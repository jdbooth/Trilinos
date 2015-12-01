
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

/** \file shylu.hpp
    
    \brief Main header file of ShyLU (Include main user calls)

    \author Siva Rajamanickam
*/
#ifndef SHYLU_HPP
#define SHYLU_HPP

// Epetra include
//#ifdef HAVE_SHYLUCORE_EPETRA
#include "Epetra_CrsMatrix.h" 
#include "Epetra_Map.h" 
#include "Epetra_MultiVector.h" 
#include "Epetra_LinearProblem.h" 
#include "Epetra_SerialComm.h"
//#endif

// Tpetra includes
#ifdef HAVE_SHYLUCORE_TPETRA
#include <Tpetra_CrsMatrix_decl.hpp>
#include <Tpetra_CrsMatrix_def.hpp>
#endif


// Amesos includes
#include "Amesos_BaseSolver.h"

// Ifpack includes
#include "Ifpack.h"
#include "Ifpack_Preconditioner.h"

// AztecOO includes
#include "AztecOO.h"

// Isorropia includes
#include "Isorropia_EpetraProber.hpp"

// Amesos2 includes
#ifdef HAVE_SHYLUCORE_AMESOS2
#include <Amesos2.hpp>
#endif


// Zoltan2 includes
#ifdef HAVE_SHYLUCORE_ZOLTAN2
#include <Zoltan2_XpetraCrsMatrixAdapter.hpp>
#include <Zoltan2_XpetraMultiVectorAdapter.hpp>
#include <Zoltan2_PartitioningProblem.hpp>
#endif


// Shylu includes
#include "shylu_symbolic.hpp"
#include "shylu_config.hpp"
#include "shylu_probing_operator.h"
#include "shylu_amesos_schur_operator.h"

#include <IQRSolver.h>



//-----------------These are the new interfaces and will replace the old--//
//Epetra/Tpetra Map
#include "shylu_type_map.hpp"

//Direct Solver Interface
#include "shylu_directsolver_interface_decl.hpp"
//#include "shylu_directsolver_interface_def.hpp"

//Parition Interface
//#include "shylu_partition_interface_decl.hpp"
//#include "shylu_partition_interface_def.hpp"


//#include "shylu_debug_manager.hpp"

#define MIN(a, b) (((a) < (b)) ? a : b)
#define MAX(a, b) (((a) > (b)) ? a : b)

/** \brief Main data structure holding needed offset and temp variables
 *
 * This structur contains ...  
 */
//Removing the typdef struct .............WE ARE USING C++ NOT C

template <class Matrix, class Vector>
struct shylu_data
{

  typedef ShyLUStackMap<Matrix,Vector>         slu_map;
  typedef typename slu_map::MT                 MT;
  typedef typename slu_map::VT                 VT;
  typedef typename slu_map::ST                 ST;
  typedef typename slu_map::LT                 LT;
  typedef typename slu_map::GT                 GT;
  typedef typename slu_map::MAP_TYPE           MAP_TYPE;
  typedef typename slu_map::GRAPH              GRAPH;
  typedef typename slu_map::IMPORT_TYPE        IMPORT_TYPE;
  typedef typename slu_map::EXPORT_TYPE        EXPORT_TYPE;


  LT Dnr;                
  //int Dnr;                    // #local rows
  LT Dnc;
  //int Dnc;                    // #local cols
  LT Snr;
  //int Snr;                    // #remote rows
  LT *DRowElems;
  //int *DRowElems;             // local rows
  LT *SRowElems;
  //int *SRowElems;             // remote rows
  LT *DColElems;
  //int *DColElems;             // Columns in D
  LT *gvals;
  // int *gvals;                 // O(n) array differentiating local/global
  //  row/col


  /*----------These all need updated--------------*/
  //Epetra_SerialComm *SComm;   // Serial comm for block diagonals

  MAP_TYPE LDRowMap;
  //Teuchos::RCP<Epetra_Map> LDRowMap;       // RowMap for block diagonals
  MAP_TYPE LGRowMap;
  //Teuchos::RCP<Epetra_Map> LGRowMap;       // RowMap for G (local)
  MAP_TYPE GMap;
  //Teuchos::RCP<Epetra_Map> GMap;           // Dist Map for G
  

  IMPORT_TYPE   BdImporter;
  //Teuchos::RCP<Epetra_Import> BdImporter;
  IMPORT_TYPE   DistImporter;
  //Teuchos::RCP<Epetra_Import> DistImporter;
  IMPORT_TYPE   BsImporter;
  //Teuchos::RCP<Epetra_Import> BsImporter;
  IMPORT_TYPE   XsImporter;
  //Teuchos::RCP<Epetra_Import> XsImporter;
  EXPORT_TYPE   XdExporter;
  //Teuchos::RCP<Epetra_Export> XdExporter;
  EXPORT_TYPE   XsExporter;
  //Teuchos::RCP<Epetra_Export> XsExporter;
  
 
  Teuchos::RCP<Vector> localrhs;
  //Teuchos::RCP<Epetra_MultiVector> localrhs;
  Teuchos::RCP<Vector> temp1;
  //Teuchos::RCP<Epetra_MultiVector> temp1;
  Teuchos::RCP<Vector> temp2;
  //Teuchos::RCP<Epetra_MultiVector> temp2;
  Teuchos::RCP<Vector> Bs;
  //Teuchos::RCP<Epetra_MultiVector> Bs;
  Teuchos::RCP<Vector> Xs;
  //Teuchos::RCP<Epetra_MultiVector> Xs;
  Teuchos::RCP<Vector> LocalXs;
  //Teuchos::RCP<Epetra_MultiVector> LocalXs;
  Teuchos::RCP<Vector> temp3;
  //Teuchos::RCP<Epetra_MultiVector> temp3;
  Teuchos::RCP<Vector> locallhs;
  //Teuchos::RCP<Epetra_MultiVector> locallhs;
  
  // temp timers
  //Teuchos::RCP<Teuchos::Time> importExportTime;
  //Teuchos::RCP<Teuchos::Time> innerIterTime;
  //Teuchos::RCP<Teuchos::Time> fwdTime;
  //Teuchos::RCP<Teuchos::Time> amesosSchurTime;
  
  //Epetra_CrsMatrix *D;        // Actual D Matrix, not reqd for Amesos_KLU
  // but required for Amesos_Pardiso
  Teuchos::RCP<IQR::IQRSolver> iqrSolver; // Solver object for IQR func.
  Teuchos::RCP<Matrix> Sbar; // Approx Schur complement
  //Teuchos::RCP<Epetra_CrsMatrix> Sbar; // Approx Schur complement
  
  Teuchos::RCP<GRAPH> localSbargraph; // graph of local Sbar
  //Teuchos::RCP<Epetra_CrsGraph> localSbargraph; // graph of local Sbar
  
  
  AztecOO *innersolver;            // inner solver
  //IterativeSolverInterface<Epetra_CrsMatrix, Epetra_MultiVector>  *innersolver;



  Teuchos::RCP<Vector> Sbarlhs;
  Teuchos::RCP<Vector> Sbarrhs;
  Teuchos::RCP<Epetra_LinearProblem> LP2;   // Local problem to solve
  Teuchos::RCP<Epetra_LinearProblem> OrigLP2;   // Local problem to solve D
  Teuchos::RCP<EpetraExt::ViewTransform<Epetra_LinearProblem> > ReIdx_LP2;


  Amesos_BaseSolver *dsolver;  // Local Subdomain solver
  //DirectSolverInterface<Epetra_CrsMatrix, Epetra_MultiVector> *dsolver;
  
  Teuchos::RCP<Ifpack_Preconditioner> schur_prec;
  Teuchos::RCP<ShyLU_Probing_Operator> schur_op;
  int lmax;                    // May be this is optimizing too much
  int rmax;                    // May be this is optimizing too much
  Teuchos::RCP<Isorropia::Epetra::Prober> guided_prober;  // Guided Prober for Sbar
  int num_compute;            // # of times Compute() has been called before
};

/*
template <>
struct shylu_data<Epetra_CrsMatrix,Epetra_MultiVector>
{
    int Dnr;                    // #local rows
    int Dnc;                    // #local cols
    int Snr;                    // #remote rows
    int *DRowElems;             // local rows
    int *SRowElems;             // remote rows
    int *DColElems;             // Columns in D
    int *gvals;                 // O(n) array differentiating local/global
                                //  row/col
    //Epetra_SerialComm *SComm;   // Serial comm for block diagonals
    Teuchos::RCP<Epetra_Map> LDRowMap;       // RowMap for block diagonals
    Teuchos::RCP<Epetra_Map> LGRowMap;       // RowMap for G (local)
    Teuchos::RCP<Epetra_Map> GMap;           // Dist Map for G

    Teuchos::RCP<Epetra_Import> BdImporter;
    Teuchos::RCP<Epetra_Import> DistImporter;
    Teuchos::RCP<Epetra_Import> BsImporter;
    Teuchos::RCP<Epetra_Import> XsImporter;
    Teuchos::RCP<Epetra_Export> XdExporter;
    Teuchos::RCP<Epetra_Export> XsExporter;

    Teuchos::RCP<Epetra_MultiVector> localrhs;
    Teuchos::RCP<Epetra_MultiVector> temp1;
    Teuchos::RCP<Epetra_MultiVector> temp2;
    Teuchos::RCP<Epetra_MultiVector> Bs;
    Teuchos::RCP<Epetra_MultiVector> Xs;
    Teuchos::RCP<Epetra_MultiVector> LocalXs;
    Teuchos::RCP<Epetra_MultiVector> temp3;
    Teuchos::RCP<Epetra_MultiVector> locallhs;

    // temp timers
    //Teuchos::RCP<Teuchos::Time> importExportTime;
    //Teuchos::RCP<Teuchos::Time> innerIterTime;
    //Teuchos::RCP<Teuchos::Time> fwdTime;
    //Teuchos::RCP<Teuchos::Time> amesosSchurTime;

    //Epetra_CrsMatrix *D;        // Actual D Matrix, not reqd for Amesos_KLU
                                // but required for Amesos_Pardiso
    Teuchos::RCP<IQR::IQRSolver> iqrSolver; // Solver object for IQR func.
    Teuchos::RCP<Epetra_CrsMatrix> Sbar; // Approx Schur complement
    Teuchos::RCP<Epetra_CrsGraph> localSbargraph; // graph of local Sbar


  AztecOO *innersolver;            // inner solver
  //IterativeSolverInterface<Epetra_CrsMatrix, Epetra_MultiVector>  *innersolver;



  Teuchos::RCP<Epetra_MultiVector> Sbarlhs;
  Teuchos::RCP<Epetra_MultiVector> Sbarrhs;
  Teuchos::RCP<Epetra_LinearProblem> LP2;   // Local problem to solve
  Teuchos::RCP<Epetra_LinearProblem> OrigLP2;   // Local problem to solve D
  Teuchos::RCP<EpetraExt::ViewTransform<Epetra_LinearProblem> > ReIdx_LP2;


  Amesos_BaseSolver *dsolver;  // Local Subdomain solver
  //DirectSolverInterface<Epetra_CrsMatrix, Epetra_MultiVector> *dsolver;
  
  Teuchos::RCP<Ifpack_Preconditioner> schur_prec;
  Teuchos::RCP<ShyLU_Probing_Operator> schur_op;
  int lmax;                    // May be this is optimizing too much
  int rmax;                    // May be this is optimizing too much
  Teuchos::RCP<Isorropia::Epetra::Prober> guided_prober;  // Guided Prober for Sbar
  int num_compute;            // # of times Compute() has been called before
                                // or in otherwords #nonlinear iteration-1
};
*/


/*----------------------OLD ------------------------
typedef struct
{
    int Dnr;                    // #local rows
    int Dnc;                    // #local cols
    int Snr;                    // #remote rows
    int *DRowElems;             // local rows
    int *SRowElems;             // remote rows
    int *DColElems;             // Columns in D
    int *gvals;                 // O(n) array differentiating local/global
                                //  row/col
    //Epetra_SerialComm *SComm;   // Serial comm for block diagonals
    Teuchos::RCP<Epetra_Map> LDRowMap;       // RowMap for block diagonals
    Teuchos::RCP<Epetra_Map> LGRowMap;       // RowMap for G (local)
    Teuchos::RCP<Epetra_Map> GMap;           // Dist Map for G

    Teuchos::RCP<Epetra_Import> BdImporter;
    Teuchos::RCP<Epetra_Import> DistImporter;
    Teuchos::RCP<Epetra_Import> BsImporter;
    Teuchos::RCP<Epetra_Import> XsImporter;
    Teuchos::RCP<Epetra_Export> XdExporter;
    Teuchos::RCP<Epetra_Export> XsExporter;

    Teuchos::RCP<Epetra_MultiVector> localrhs;
    Teuchos::RCP<Epetra_MultiVector> temp1;
    Teuchos::RCP<Epetra_MultiVector> temp2;
    Teuchos::RCP<Epetra_MultiVector> Bs;
    Teuchos::RCP<Epetra_MultiVector> Xs;
    Teuchos::RCP<Epetra_MultiVector> LocalXs;
    Teuchos::RCP<Epetra_MultiVector> temp3;
    Teuchos::RCP<Epetra_MultiVector> locallhs;

    // temp timers
    //Teuchos::RCP<Teuchos::Time> importExportTime;
    //Teuchos::RCP<Teuchos::Time> innerIterTime;
    //Teuchos::RCP<Teuchos::Time> fwdTime;
    //Teuchos::RCP<Teuchos::Time> amesosSchurTime;

    //Epetra_CrsMatrix *D;        // Actual D Matrix, not reqd for Amesos_KLU
                                // but required for Amesos_Pardiso
    Teuchos::RCP<IQR::IQRSolver> iqrSolver; // Solver object for IQR func.
    Teuchos::RCP<Epetra_CrsMatrix> Sbar; // Approx Schur complement
    Teuchos::RCP<Epetra_CrsGraph> localSbargraph; // graph of local Sbar


  AztecOO *innersolver;            // inner solver
  //IterativeSolverInterface<Epetra_CrsMatrix, Epetra_MultiVector>  *innersolver;



  Teuchos::RCP<Epetra_MultiVector> Sbarlhs;
  Teuchos::RCP<Epetra_MultiVector> Sbarrhs;
  Teuchos::RCP<Epetra_LinearProblem> LP2;   // Local problem to solve
  Teuchos::RCP<Epetra_LinearProblem> OrigLP2;   // Local problem to solve D
  Teuchos::RCP<EpetraExt::ViewTransform<Epetra_LinearProblem> > ReIdx_LP2;


  Amesos_BaseSolver *dsolver;  // Local Subdomain solver
  //DirectSolverInterface<Epetra_CrsMatrix, Epetra_MultiVector> *dsolver;
  
  Teuchos::RCP<Ifpack_Preconditioner> schur_prec;
  Teuchos::RCP<ShyLU_Probing_Operator> schur_op;
  int lmax;                    // May be this is optimizing too much
  int rmax;                    // May be this is optimizing too much
  Teuchos::RCP<Isorropia::Epetra::Prober> guided_prober;  // Guided Prober for Sbar
  int num_compute;            // # of times Compute() has been called before
                                // or in otherwords #nonlinear iteration-1
} shylu_data;

*/

/** \brief Main function call into ShylU
 *
 * How to use?
 */

template <class Matrix, class Vector>
    int shylu_factor(Matrix *A, shylu_symbolic<Matrix,Vector> *ssym, shylu_data<Matrix,Vector> *data, shylu_config<Matrix,Vector> *config);


/*---------------OLD----------------------
  int shylu_factor(Epetra_CrsMatrix *A, shylu_symbolic *ssym, shylu_data *data,
                 shylu_config *config);
*/


/** \brief Call symbolic factorization on matrix
 *
 */
template <class Matrix, class Vector>
  int shylu_symbolic_factor(Matrix *A, shylu_symbolic<Matrix,Vector> *ssym, shylu_data<Matrix,Vector> *data, shylu_config<Matrix,Vector> *config);


  /*-----------------OLD-----------------
int shylu_symbolic_factor
(
    Epetra_CrsMatrix *A,    // i/p: A matrix
    shylu_symbolic *ssym,   // symbolic structure
    shylu_data *data,       // numeric structure, TODO: Required ?
    shylu_config *config   // i/p: library configuration
);

  */

/** \brief Call solve on multiple RHS
 *
 */


  template <class Matrix, class Vector>
  int shylu_solve(shylu_symbolic<Matrix,Vector> *ssym, shylu_data<Matrix,Vector> *data, shylu_config<Matrix,Vector> *config,
                  const Vector &X, Vector &Y);
   /*---------------------OLD-------------------
int shylu_solve(shylu_symbolic *ssym, shylu_data *data, shylu_config *config,
    const Epetra_MultiVector& X, Epetra_MultiVector& Y);
   */


/** \brief Compute an approximate Schur Complement (Narrow Sep)
 *
 *  Computate an approximate Schur Complement either using ...
 */ 





/* COME BACK... need whole rewirte
template <class Matrix, class Vector>
  Teuchos::RCP<Matrix> computeApproxSchur(shylu_config *config,
                                          shylu_symbolic *ssym,
*/              


Teuchos::RCP<Epetra_CrsMatrix> computeApproxSchur(shylu_config<Epetra_CrsMatrix,Epetra_MultiVector> *config,
                                                  shylu_symbolic<Epetra_CrsMatrix,Epetra_MultiVector> *ssym,
    Epetra_CrsMatrix *G, Epetra_CrsMatrix *R,
    Epetra_LinearProblem *LP, Amesos_BaseSolver *solver,
    Ifpack_Preconditioner *ifSolver, Epetra_CrsMatrix *C,
    Epetra_Map *localDRowMap);

/** \brief Compute an approximate Shur Complete (Wide Sep)
 *
 * Compute an approximate Schur Complement based on a wide seperator.
 * Options include ...
 */
Teuchos::RCP<Epetra_CrsMatrix> computeApproxWideSchur(
                                                      shylu_config<Epetra_CrsMatrix, Epetra_MultiVector> *config,
                                                      shylu_symbolic<Epetra_CrsMatrix,Epetra_MultiVector> *ssym,   // symbolic structure
    Epetra_CrsMatrix *G, Epetra_CrsMatrix *R,
    Epetra_LinearProblem *LP, Amesos_BaseSolver *solver,
    Ifpack_Preconditioner *ifSolver, Epetra_CrsMatrix *C,
    Epetra_Map *localDRowMap);

/** \brief Compute an approximate Schur Complement using the option of Guided Probing
 *
 *  Compute an approximate Schur Complement based on probing of important nonzero values.
 */
Teuchos::RCP<Epetra_CrsMatrix> computeSchur_GuidedProbing
(
 shylu_config<Epetra_CrsMatrix,Epetra_MultiVector> *config,
 shylu_symbolic<Epetra_CrsMatrix,Epetra_MultiVector> *ssym,   // symbolic structure
 shylu_data<Epetra_CrsMatrix, Epetra_MultiVector> *data,       // numeric structure
    Epetra_Map *localDRowMap
);
#endif // SHYLU_HPP
