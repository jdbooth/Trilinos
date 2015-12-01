
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

/** \file shylu_util.cpp

    \brief Utilities for ShyLU

    \author Siva Rajamanickam

*/
#ifndef SHYLU_UTIL_DEF_HPP
#define SHYLU_UTIL_DEF_HPP

#include <assert.h>
#include <fstream>

#include "shylu_util_decl.hpp"
//#include "shylu_type_map.hpp"


//Epetra
//#ifdef HAVE_SHYLUCORE_EPETRA
#ifdef HAVE_SHYLUCORE_MPI
#include "Epetra_MpiComm.h"
#else
#include "Epetra_SerialComm.h"
#endif //HAVE_SHYLUCORE_MPI

#include "Epetra_CrsMatrix.h"
#include "EpetraExt_RowMatrixOut.h"
//#endif // HAVE_SHYLUCORE_EPETRA


//Tpetra
#ifdef HAVE_SHYLUCORE_TPETRA
#include <Tpetra_CrsMatrix_decl.hpp>
#include <Tpetra_CrsMatrix_def.hpp>
#include <Tpetra_MultiVector_decl.hpp>
#include <Tpetra_MultiVector_def.hpp>
#endif


//Teuchos includes
#include "Teuchos_XMLParameterListHelpers.hpp"

//Isorropia includes
//#ifdef HAVE_SHYLUCORE_ISORRPIA
#include "Isorropia_Epetra.hpp"
#include "Isorropia_EpetraRedistributor.hpp"
#include "Isorropia_EpetraPartitioner.hpp"
//#endif // HAVE_SHYLUCORE_ISORRPIA

using namespace std;

/*-----------------------balanceAndRedistribute------------------*/
//This is the key partition/redistribution function call
// Currently takes only MpiComm
template <class Matrix, class Vector>
Matrix* 
balanceAndRedistribute
(
 Matrix *A, 
 Teuchos::ParameterList isoList
)
{
  cout << "place new function here" << endl;
  cout << "empty" << endl;
  return new Matrix();
}//end balanceAndRedistribute<Matrix,Vector>

//needed because cpp file
//This is not going to work as we will have to template on everything!!!
#ifdef HAVE_SHYLUCORE_TPETRA
//template Tpetra::CrsMatrix* balanceAndRedistribute<Tpetra::CrsMatrix,Tpetra::MultiVector>(Tpetra::CrsMatrix, Teuchos::ParameterList);
#endif

//#ifdef HAVE_SHLYLUCORE_EPETRA
template <>
Epetra_CrsMatrix* 
balanceAndRedistribute<Epetra_CrsMatrix, Epetra_MultiVector>
(
 Epetra_CrsMatrix *A,
 Teuchos::ParameterList isoList
)
{
    // int myPID = A->Comm().MyPID(); // unused

    // Debug [
    Epetra_Map ARowMap = A->RowMap();
    // int nrows = ARowMap.NumMyElements(); // unused
    // int *rows = ARowMap.MyGlobalElements(); // unused
    // ]

    // ==================== Symbolic factorization =========================
    // 1. Partition and redistribute [
    Isorropia::Epetra::Partitioner *partitioner = new
                            Isorropia::Epetra::Partitioner(A, isoList, false);
    partitioner->partition();

    Isorropia::Epetra::Redistributor rd(partitioner);
    Epetra_CrsMatrix *newA;
    rd.redistribute(*A, newA);
    // ]
    EpetraExt::RowMatrixToMatlabFile("A.mat", *newA);

    delete partitioner;
    return newA;
}
//#endif // HAVE_SHYLUCORE_EPETRA

/*------------------------------checkMaps----------------------*/
/* TODO : Do this only for Debug ? */

template <class Matrix, class Vector>
void 
checkMaps(Matrix *A)
{
  typedef ShyLUStackMap<Matrix,Vector>    slu_map;
  typedef typename slu_map::GT            GT;
  typedef typename slu_map::GT_ARRAY      GT_ARRAY;
  typedef typename slu_map::MAP_TYPE      MAP_TYPE;
 
  // Get column map
  MAP_TYPE AColMap = slu_map::ColMap(A);
  // Get domain map
  MAP_TYPE ADomainMap = slu_map::DomainMap(A);  
  GT  nelems = slu_map::NumMyElements(ADomainMap);

#ifndef NDEBUG
  // mfh 25 May 2015: Only used in an assert() below.
  // assert() is defined to nothing in a release build.
  GT_ARRAY dom_cols = slu_map::MyGlobalElements(ADomainMap);

#endif // NDEBUG

  // Get range map
  MAP_TYPE ARangeMap = slu_map::RangeMap(A);

#ifndef NDEBUG
  // mfh 25 May 2015: Only used in an assert() below.
  // assert() is defined to nothing in a release build.
  GT_ARRAY ran_cols = slu_map::MyGlobalElements(ARangeMap);

#endif // NDEBUG
  
  // Get row map
  MAP_TYPE ARowMap = slu_map::RowMap(A);

#ifndef NDEBUG
  // mfh 25 May 2015: Only used in an assert() below.
  // assert() is defined to nothing in a release build.
  GT_ARRAY rows = slu_map::MyGlobalElements(ARowMap);

#endif // NDEBUG
  
  //cout <<"In PID ="<< A->Comm().MyPID() <<" #cols="<< ncols << " #rows="<<
  //nrows <<" #domain elems="<< nelems <<" #range elems="<< npts << endl;
  // See if domain map == range map == row map
  for (int i = 0; i < nelems ; i++)
    {
      // Will this always be the case ? We will find out if assertion fails !
      assert(dom_cols[i] == ran_cols[i]);
      assert(rows[i] == ran_cols[i]);
    }
}//end checkMaps<Matrix,Vector>

/*-------------------------------findLocalColumns-----------------*/
// TODO: SNumGlobalCols never used

template <class Matrix, class Vector>
void findLocalColumns
(
 Matrix *A,
 int *gvals,
 int &SNumGlobalCols
 )
{
  
  typedef ShyLUStackMap<Matrix,Vector>  slu_map;
  typedef typename slu_map::GT          GT;
  typedef typename slu_map::GT_ARRAY    GT_ARRAY;
  typedef typename slu_map::MAP_TYPE    MAP_TYPE;
  

  GT n = slu_map::NumGlobalRows(A);

  // Get column map
  MAP_TYPE  AColMap = slu_map::ColMap(A);

  GT ncols = slu_map::NumMyElements(AColMap);

  GT_ARRAY cols = slu_map::MyGlobalElements(AColMap);

  
  // 2. Find column permutation [
  // Find all columns in this proc
  //Note: do we want to make vals of map type??
   int *vals = new int[n];       // vector of size n, not ncols
   for (int i = 0; i < n ; i++)
    {
      vals[i] = 0;
      gvals[i] = 0;
    }
  
  // Null columns in A are not part of any proc
  for (int i = 0; i < ncols ; i++)
    {
      vals[cols[i]] = 1;        // Set to 1 for locally owned columns
    }
  
  // Bottleneck?: Compute the column permutation
  slu_map::SumAll(A,vals,gvals,n);
  
  SNumGlobalCols = 0;
  for (int i = 0; i < n ; i++)
    {
      //cout << gvals[i] ;
      if (gvals[i] > 1)
        SNumGlobalCols++;
    }
  //cout << endl;
  //cout << "Snum Global cols=" << SNumGlobalCols << endl;
  
  delete[] vals;
  
  return;
}//end findLocalColumns<Matrix,Vector>

/*------------------------------findNarrowSeparator----------------*/
// This function uses a very simple tie-breaking heuristic to find a
// "narrow" separator from a wide separator. The vertices in the proc with
// smaller procID will become part of the separator
// This is not a true narrow separator, which needs a vertex cover algorithm.
// This is like a medium separator !
// TODO : This assumes symmetry I guess, Check

template <class Matrix, class Vector>
void findNarrowSeparator
(
 Matrix *A, 
 int *gvals
 )
{
  
  typedef ShyLUStackMap<Matrix,Vector> slu_map;
  typedef typename slu_map::GT         GT;
  typedef typename slu_map::ST         ST;
  typedef typename slu_map::GT_ARRAY   GT_ARRAY;
  typedef typename slu_map::ST_ARRAY   ST_ARRAY;
  typedef typename slu_map::MAP_TYPE   MAP_TYPE;
  
  GT  nentries;

  ST_ARRAY values;
  GT_ARRAY indices;
  
  GT n = slu_map::NumGlobalRows(A);

  int myPID = slu_map::MyPID(A);

  // Get row map
  MAP_TYPE rMap = slu_map::RowMap(A);
  MAP_TYPE cMap = slu_map::ColMap(A);
  GT_ARRAY  rows = slu_map::MyGlobalElements(rMap);
  GT      relems = slu_map::NumMyElements(rMap);

  //Do we want to change these a map type??
  int *vals = new int[n];       // vector of size n, not ncols
  int *allGIDs = new int[n];       // vector of size n, not ncols
  for (int i = 0; i < n ; i++) // initialize to zero
    {
      vals[i] = 0;
    }
  
  // Rows are uniquely owned, so this will work
  for (int i = 0; i < relems ; i++)
    {
      vals[rows[i]] = myPID;        // I own relems[i]
    }

  
  // **************** Collective communication **************
  // This will not scale well for very large number of nodes
  // But on the node this should be fine
  slu_map::SumAll(A,vals,allGIDs,n);

  
  // At this point all procs know who owns what rows
  for (int i = 0; i < n ; i++) // initialize to zero
    vals[i] = 0;

  GT  gid, cgid;
  for (int i = 0; i < relems; i++)
    {
      gid = rows[i];
      //cout << "PID=" << myPID << " " << "rowid=" << gid ;
      if (gvals[gid] != 1)
        {
          //cout << " in the sep ";
          bool movetoBlockDiagonal = false;
          // mfh 25 May 2015: This call used to assign its (int)
          // return value to 'err'.  I got rid of this, because
          // 'err' was unused.  This resulted in a "set but unused
          // variable" warning.
                  
          slu_map::ExtractMyRowView(A,i,&nentries,values,indices);
          //cout << " with nentries= "<< nentries;

          assert(nentries != 0);
          for (int j = 0; j < nentries; j++)
            {

              cgid = slu_map::GID(cMap, indices[j]);
     
              assert(cgid != -1);
              if (gvals[cgid] == 1 || allGIDs[cgid] == myPID)
                continue; // simplify the rest

             
              if (allGIDs[cgid] < myPID)
                {
                  // row cgid is owned by a proc with smaller PID
                  movetoBlockDiagonal = true;
                  //cout << "\t mving to diag because of column" << cgid;
                }
              else
                {
                  // There is at least one edge from this vertex to a
                  // vertex in a proc with PID > myPID, cannot move
                  // to diagonal. This is too restrictive, but
                  // important for correctness, until we can use a
                  // vertex cover algorithm.
                  movetoBlockDiagonal = false;
                  break;
                  //cout << "\tNo problem with cgid=" << cgid << "in sep";
                }
             
            }
          if (movetoBlockDiagonal)
            {
              //cout << "Moving to Diagonal";
              vals[gid] = 1;
              gvals[gid] = 1; // The smaller PIDs have to know about this
              // change. Send the change using gvals.
            }
        }
      else
        {
          // do nothing, in the diagonal block already
          //cout << "In the diagonal block";
        }
      //cout << endl;
    }
  
  // Reuse allGIDs to propagate the result of moving to diagonal
  for (int i = 0; i < n ; i++) // initialize to zero
    allGIDs[i] = 0;
  
  slu_map::SumAll(A,vals,allGIDs,n);
 
  for (int i = 0; i < n ; i++)
    {
      if (allGIDs[i] == 1)
        {
          // Some interface columns will have gvals[1] after this
          // as the separator is narrow now.
          gvals[i] = 1; // GIDs as indices assumption
        }
    }
  
  //Might get memory error on vals
  delete[] vals;
  delete[] allGIDs;  
  
}//end find NarrowSeparator<Matrix,Vector>


/*-------------------------------findBlockElems--------------------*/

template <class Matrix, class Vector>
void findBlockElems
(
 Matrix *A, 
 typename ShyLUStackMap<Matrix,Vector>::GT nrows,
 //int nrows,
 typename ShyLUStackMap<Matrix,Vector>::GT *rows,
 //int *rows, 
 int *gvals, 
 int Lnr, 
 int *LeftElems,
 int Rnr,
 int *RightElems,
 string s1, string s2, bool cols)
{

  
  typedef ShyLUStackMap<Matrix,Vector>      slu_map;
  typedef typename slu_map::GT              GT;
  typedef typename slu_map::GT_ARRAY        GT_ARRAY;
  typedef typename slu_map::MAP_TYPE        MAP_TYPE;


  std::cout << "Called findBlockElems  " << cols << std::endl; 


  GT gid;
  //int gid;
  int rcnt = 0; int lcnt = 0;
  // Assemble ids in two arrays
  ostringstream ssmsg1;
  ostringstream ssmsg2;
  
#ifdef DUMP_MATRICES
  ostringstream fnamestr;
  fnamestr << s1 << ".mat";
  string fname = fnamestr.str();
  ofstream outD(fname.c_str());
    
  ostringstream fnamestrR;
  fnamestrR << s2 << ".mat";
  string fnameR = fnamestrR.str();
  ofstream outR(fnameR.c_str());
#endif
    
  ssmsg1 << s1;
  ssmsg2 << s2;
  for (int i = 0; i < nrows; i++)
    {
      gid = rows[i];
      assert (gvals[gid] >= 1);
      // If the row is local & row/column is not shared then row/column
      // belongs to D (this is not true for R, which can have more columns
      // than D)
      //if (A->LRID(gid) != -1 && gvals[gid] == 1)
      if (gvals[gid] == 1)
        {
          if (cols && A->LRID(gid) == -1) continue;
          assert(lcnt < Lnr);
          LeftElems[lcnt++] = gid;
          ssmsg1 << gid << " ";
#ifdef DUMP_MATRICES
          outD << gid << endl;
#endif
        }
      else
        {
          assert(rcnt < Rnr);
          RightElems[rcnt++] = gid;
          ssmsg2 << gid << " ";
#ifdef DUMP_MATRICES
          outR << gid << endl;
#endif
        }
    }
  
#ifdef DUMP_MATRICES
  outD.close();
  outR.close();
#endif
  
#ifdef DEBUG
  cout << ssmsg1.str() << endl;
  cout << ssmsg2.str() << endl;
#endif
  ssmsg1.clear(); ssmsg1.str("");
  ssmsg2.clear(); ssmsg2.str("");
  
  assert(lcnt == Lnr);
  assert(rcnt == Rnr);
  
}//end findBlockElems<Matrix,Vector>

#endif //SHYLU_UTIl_DEF_HPP
