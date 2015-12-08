// @HEADER
//
// ***********************************************************************
//
//        MueLu: A package for multigrid based preconditioning
//                  Copyright 2012 Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact
//                    Jonathan Hu       (jhu@sandia.gov)
//                    Andrey Prokopenko (aprokop@sandia.gov)
//                    Ray Tuminaro      (rstumin@sandia.gov)
//
// ***********************************************************************
//
// @HEADER
#include <Teuchos_UnitTestHarness.hpp>
#include <Teuchos_ScalarTraits.hpp>

#include <Xpetra_MultiVectorFactory.hpp>
#include <Xpetra_MatrixMatrix.hpp>

#include <MueLu_config.hpp>

#include <MueLu_TestHelpers.hpp>
#include <MueLu_Version.hpp>

#include <MueLu_Utilities.hpp>

// This file is intended to house all the tests for MueLu_Utilities.hpp.

namespace MueLuTests {


  TEUCHOS_UNIT_TEST_TEMPLATE_4_DECL(Utilities,MatMatMult_EpetraVsTpetra,Scalar,LocalOrdinal,GlobalOrdinal,Node)
  {
#if defined(HAVE_MUELU_TPETRA) && defined(HAVE_MUELU_EPETRAEXT)
#   include <MueLu_UseShortNames.hpp>
    MUELU_TESTING_SET_OSTREAM;
    out << "version: " << MueLu::Version() << std::endl;
    out << "This test compares the matrix matrix multiply between Tpetra and Epetra" << std::endl;

    MUELU_TESTING_LIMIT_EPETRA_SCOPE_TPETRA_IS_DEFAULT(Scalar,GlobalOrdinal,Node);

    RCP<const Teuchos::Comm<int> > comm = Parameters::getDefaultComm();

    typedef typename Teuchos::ScalarTraits<Scalar>::magnitudeType magnitude_type;

    //Calculate result = (Op*Op)*X for Epetra
    GO nx = 37*comm->getSize();
    GO ny = nx;
    RCP<Matrix> Op = TestHelpers::TestFactory<Scalar, LocalOrdinal, GlobalOrdinal, Node>::Build2DPoisson(nx,ny,Xpetra::UseEpetra);
    RCP<Matrix> OpOp = Xpetra::MatrixMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>::Multiply(*Op,false,*Op,false,out);
    RCP<MultiVector> result = MultiVectorFactory::Build(OpOp->getRangeMap(),1);
    RCP<MultiVector> X = MultiVectorFactory::Build(OpOp->getDomainMap(),1);
    Teuchos::Array<magnitude_type> xnorm(1);
    X->setSeed(8675309);
    X->randomize(true);
    X->norm2(xnorm);
    OpOp->apply(*X,*result,Teuchos::NO_TRANS,(Scalar)1.0,(Scalar)0.0);
    Teuchos::Array<magnitude_type> normEpetra(1);
    result->norm2(normEpetra);

    // aid debugging by calculating Op*(Op*X)
    RCP<MultiVector> workVec = MultiVectorFactory::Build(OpOp->getRangeMap(),1);
    RCP<MultiVector> check1 = MultiVectorFactory::Build(OpOp->getRangeMap(),1);
    Op->apply(*X,*workVec,Teuchos::NO_TRANS,(Scalar)1.0,(Scalar)0.0);
    Op->apply(*workVec,*check1,Teuchos::NO_TRANS,(Scalar)1.0,(Scalar)0.0);
    Teuchos::Array<magnitude_type> normCheck1(1);
    check1->norm2(normCheck1);

    //Calculate result = (Op*Op)*X for Tpetra
    Op = TestHelpers::TestFactory<Scalar, LocalOrdinal, GlobalOrdinal, Node>::Build2DPoisson(nx,ny,Xpetra::UseTpetra);
    OpOp = Xpetra::MatrixMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>::Multiply(*Op,false,*Op,false,out);
    result = MultiVectorFactory::Build(OpOp->getRangeMap(),1);
    X = MultiVectorFactory::Build(OpOp->getDomainMap(),1);
    X->setSeed(8675309);
    X->randomize(true);
    X->norm2(xnorm);
    OpOp->apply(*X,*result,Teuchos::NO_TRANS,(Scalar)1.0,(Scalar)0.0);
    Teuchos::Array<magnitude_type> normTpetra(1);
    result->norm2(normTpetra);

    // aid debugging by calculating Op*(Op*X)
    workVec = MultiVectorFactory::Build(OpOp->getRangeMap(),1);
    RCP<MultiVector> check2 = MultiVectorFactory::Build(OpOp->getRangeMap(),1);
    Op->apply(*X,*workVec,Teuchos::NO_TRANS,(Scalar)1.0,(Scalar)0.0);
    Op->apply(*workVec,*check2,Teuchos::NO_TRANS,(Scalar)1.0,(Scalar)0.0);
    Teuchos::Array<magnitude_type> normCheck2(1);
    check2->norm2(normCheck2);

    TEST_FLOATING_EQUALITY(normEpetra[0], normTpetra[0], 1e-12);
    out << "Epetra ||A*(A*x)|| = " << normCheck1[0] << std::endl;
    out << "Tpetra ||A*(A*x)|| = " << normCheck2[0] << std::endl;
#   else
    out << "Skipping test because some required packages are not enabled (Tpetra, EpetraExt)." << std::endl;
#   endif

  } //EpetraVersusTpetra

  TEUCHOS_UNIT_TEST_TEMPLATE_4_DECL(Utilities,DetectDirichletRows,Scalar,LocalOrdinal,GlobalOrdinal,Node)
  {
#   include <MueLu_UseShortNames.hpp>
    MUELU_TESTING_SET_OSTREAM;
    MUELU_TESTING_LIMIT_EPETRA_SCOPE(Scalar,GlobalOrdinal,Node);

    typedef typename Teuchos::ScalarTraits<Scalar> TST;

    RCP<Matrix> A = TestHelpers::TestFactory<Scalar, LocalOrdinal, GlobalOrdinal, Node>::Build1DPoisson(100);
    Teuchos::ArrayView<const LocalOrdinal> indices;
    Teuchos::ArrayView<const Scalar>  values;

    LocalOrdinal localRowToZero = 5;
    A->resumeFill();
    A->getLocalRowView(localRowToZero, indices, values);
    Array<Scalar> newvalues(values.size(),TST::zero());
    for (int j = 0; j < indices.size(); j++)
      //keep diagonal
      if (indices[j] == localRowToZero) newvalues[j] = values[j];
    A->replaceLocalValues(localRowToZero,indices,newvalues);

    A->fillComplete();

    ArrayRCP<const bool> drows = Utilities::DetectDirichletRows(*A);
    TEST_EQUALITY(drows[localRowToZero], true);
    TEST_EQUALITY(drows[localRowToZero-1], false);

    A->resumeFill();
    A->getLocalRowView(localRowToZero, indices, values);
    for (int j = 0; j < indices.size(); j++)
      //keep diagonal
      if (indices[j] == localRowToZero) newvalues[j] = values[j];
      else newvalues[j] = Teuchos::as<Scalar>(0.25);
    A->replaceLocalValues(localRowToZero,indices,newvalues);

    //row 5 should not be Dirichlet
    drows = Utilities::DetectDirichletRows(*A,TST::magnitude(0.24));
    TEST_EQUALITY(drows[localRowToZero], false);
    TEST_EQUALITY(drows[localRowToZero-1], false);

    //row 5 should be Dirichlet
    drows = Utilities::DetectDirichletRows(*A,TST::magnitude(0.26));
    TEST_EQUALITY(drows[localRowToZero], true);
    TEST_EQUALITY(drows[localRowToZero-1], false);

  } //DetectDirichletRows

#define MUELU_ETI_GROUP(Scalar, LO, GO, Node) \
         TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(Utilities,MatMatMult_EpetraVsTpetra,Scalar,LO,GO,Node) \
         TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(Utilities,DetectDirichletRows,Scalar,LO,GO,Node)

#include <MueLu_ETI_4arg.hpp>

}//namespace MueLuTests

