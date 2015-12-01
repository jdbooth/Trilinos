
//@HEADER
// ************************************************************************
// 
//               ShyLU: Hybrid preconditioner package
//                 Copyright 2012 Sandia Corporation
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
// Questions? Contact Michael A. Heroux (maherou@sandia.gov) 
// 
// ************************************************************************
//@HEADER

#ifndef SHYLU_SYMBOLIC_HPP
#define SHYLU_SYMBOLIC_HPP


//Epetra
//#ifdef HAVE_SHYLUCORE_EPETRA
#include "Epetra_CrsMatrix.h" 
#include "Epetra_Map.h" 
#include "Epetra_MultiVector.h" 
#include "Epetra_LinearProblem.h"
//#endif //HAVE_SHYLUCORE_EPETRA


//Tpetra
#ifdef HAVE_SHYLUCORE_TPETRA
#include <Tpetra_CrsMatrix_decl.hpp>
#include <Tpetra_CrsMatrix_def.hpp>
#include <Tpetra_Map_decl.hpp>
#include <Tpetra_Map_def.hpp>
#endif //HAVE_SHYLUCORE_TPETRA

//#ifdef HAVE_SHYLUCORE_ISORROPIA
#include "Isorropia_EpetraProber.hpp"
//#endif // HAVE_SHYLUCORE_ISORROPIA

//#include "EpetraExt_Transpose_RowMatrix.h"
//#ifdef HAVE_SHYLUCORE_EPETRAEXT
#include <EpetraExt_Reindex_LinearProblem2.h>
//#endif

#include "Ifpack_Preconditioner.h"

// This is NOT just the symbolic structure, needs a better name
//Changing from typedef struct to just struct......... this is C++ not C!
//Started convert
template <class Matrix, class Vector>
struct shylu_symbolic
{
  Teuchos::RCP<Matrix> D;        // D Matrix
  //Teuchos::RCP<Epetra_CrsMatrix> DT;       // D Transpose Matrix
  //Teuchos::RCP<EpetraExt::RowMatrix_Transpose> transposer;
  Teuchos::RCP<Matrix> C;        // Column separator
  Teuchos::RCP<Matrix> R;        // Row separator
  Teuchos::RCP<Matrix> G;        // G Matrix (A22 block)
  Teuchos::RCP<Epetra_LinearProblem> LP;   // Local problem to solve D
  Teuchos::RCP<Epetra_LinearProblem> OrigLP;   // Local problem to solve D
  Teuchos::RCP<EpetraExt::ViewTransform<Epetra_LinearProblem> > ReIdx_LP ;
  Teuchos::RCP<Vector> Dlhs;
  Teuchos::RCP<Vector> Drhs;
  Teuchos::RCP<Vector> Gvec;
  Teuchos::RCP<Amesos_BaseSolver> Solver;  // Local solver for D
  Teuchos::RCP<Ifpack_Preconditioner> ifSolver; //Local incomplete preconditioner
  Teuchos::RCP<Epetra_CrsGraph> Sg;        // The approximate graph of S
                                             // Graph(S) + few diagonals
  Teuchos::RCP<Isorropia::Epetra::Prober> prober;  // Prober for Sbar
};


/*
template <class Matrix, class Vector>
struct shylu_symbolic
{
  Teuchos::RCP<Epetra_CrsMatrix> D;        // D Matrix
  //Teuchos::RCP<Epetra_CrsMatrix> DT;       // D Transpose Matrix
  //Teuchos::RCP<EpetraExt::RowMatrix_Transpose> transposer;
  Teuchos::RCP<Epetra_CrsMatrix> C;        // Column separator
  Teuchos::RCP<Epetra_CrsMatrix> R;        // Row separator
  Teuchos::RCP<Epetra_CrsMatrix> G;        // G Matrix (A22 block)
  Teuchos::RCP<Epetra_LinearProblem> LP;   // Local problem to solve D
  Teuchos::RCP<Epetra_LinearProblem> OrigLP;   // Local problem to solve D
  Teuchos::RCP<EpetraExt::ViewTransform<Epetra_LinearProblem> > ReIdx_LP ;
  Teuchos::RCP<Epetra_MultiVector> Dlhs;
  Teuchos::RCP<Epetra_MultiVector> Drhs;
  Teuchos::RCP<Epetra_MultiVector> Gvec;
  Teuchos::RCP<Amesos_BaseSolver> Solver;  // Local solver for D
  Teuchos::RCP<Ifpack_Preconditioner> ifSolver; //Local incomplete preconditioner
  Teuchos::RCP<Epetra_CrsGraph> Sg;        // The approximate graph of S
                                             // Graph(S) + few diagonals
  Teuchos::RCP<Isorropia::Epetra::Prober> prober;  // Prober for Sbar
};
*/

template <>
struct shylu_symbolic<Epetra_CrsMatrix, Epetra_MultiVector>
{
  Teuchos::RCP<Epetra_CrsMatrix> D;        // D Matrix
  //Teuchos::RCP<Epetra_CrsMatrix> DT;       // D Transpose Matrix
  //Teuchos::RCP<EpetraExt::RowMatrix_Transpose> transposer;
  Teuchos::RCP<Epetra_CrsMatrix> C;        // Column separator
  Teuchos::RCP<Epetra_CrsMatrix> R;        // Row separator
  Teuchos::RCP<Epetra_CrsMatrix> G;        // G Matrix (A22 block)
  Teuchos::RCP<Epetra_LinearProblem> LP;   // Local problem to solve D
  Teuchos::RCP<Epetra_LinearProblem> OrigLP;   // Local problem to solve D
  Teuchos::RCP<EpetraExt::ViewTransform<Epetra_LinearProblem> > ReIdx_LP ;
  Teuchos::RCP<Epetra_MultiVector> Dlhs;
  Teuchos::RCP<Epetra_MultiVector> Drhs;
  Teuchos::RCP<Epetra_MultiVector> Gvec;
  Teuchos::RCP<Amesos_BaseSolver> Solver;  // Local solver for D
  Teuchos::RCP<Ifpack_Preconditioner> ifSolver; //Local incomplete preconditioner
  Teuchos::RCP<Epetra_CrsGraph> Sg;        // The approximate graph of S
  // Graph(S) + few diagonals
  Teuchos::RCP<Isorropia::Epetra::Prober> prober;  // Prober for Sbar
};//shylu_symbolic<Eptra_CrsMatrix, Epetra_MultiVector>





/*-----------------------OLD
typedef struct
{
    Teuchos::RCP<Epetra_CrsMatrix> D;        // D Matrix
    //Teuchos::RCP<Epetra_CrsMatrix> DT;       // D Transpose Matrix
    //Teuchos::RCP<EpetraExt::RowMatrix_Transpose> transposer;
    Teuchos::RCP<Epetra_CrsMatrix> C;        // Column separator
    Teuchos::RCP<Epetra_CrsMatrix> R;        // Row separator
    Teuchos::RCP<Epetra_CrsMatrix> G;        // G Matrix (A22 block)
    Teuchos::RCP<Epetra_LinearProblem> LP;   // Local problem to solve D
    Teuchos::RCP<Epetra_LinearProblem> OrigLP;   // Local problem to solve D
    Teuchos::RCP<EpetraExt::ViewTransform<Epetra_LinearProblem> > ReIdx_LP ;
    Teuchos::RCP<Epetra_MultiVector> Dlhs;
    Teuchos::RCP<Epetra_MultiVector> Drhs;
    Teuchos::RCP<Epetra_MultiVector> Gvec;
    Teuchos::RCP<Amesos_BaseSolver> Solver;  // Local solver for D
    Teuchos::RCP<Ifpack_Preconditioner> ifSolver; //Local incomplete preconditioner
    Teuchos::RCP<Epetra_CrsGraph> Sg;        // The approximate graph of S
                                             // Graph(S) + few diagonals
    Teuchos::RCP<Isorropia::Epetra::Prober> prober;  // Prober for Sbar
} shylu_symbolic;
*/

#endif // SHYLU_SYMBOLIC_HPP
