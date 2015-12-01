#ifndef SHYLU_TYPE_MAP_HPP
#define SHYLU_TYPE_MAP_HPP

//Might be good to use this mapping function in place of 
//changing everything between eptra and tpetra

#ifdef HAVE_SHYLUCORE_TPETRA
#include <Tpetra_ConfigDefs.hpp>
#include <Tpetra_CrsMatrix_decl.hpp>
#include <Tpetra_CrsMatrix_def.hpp>
#include <Tpetra_CrsGraph_decl.hpp>
#include <Tpetra_CrsGraph_def.hpp>
#include <Tpetra_Map_decl.hpp>
#include <Tpetra_Map_def.hpp>
#include <Tpetra_Import.hpp>
#include <Tpetra_Export.hpp>
#include <Teuchos_ArrayView.hpp>
#include <Teuchos_RCP.hpp>
#include <Teuchos_Comm.hpp>
#ifdef HAVE_SHYLUCORE_MPI
#include <Teuchos_DefaultMpiComm.hpp>
#else
#include <Teuchos_SerialComm.hpp>
#endif

#endif


//Big note:    May want to use Teuchos::ArrayRCP ???????


//Tpetra
template <class Matrix, class Vector>
struct ShyLUStackMap
{
#ifdef HAVE_SHYLUCORE_TPETRA
  typedef  Matrix                                   MT;
  typedef  Vector                                   VT;
  typedef  typename MT::scalar_type                 ST;
  typedef  typename MT::local_ordinal_type          LT; 
  typedef  typename MT::global_ordinal_type         GT;
  typedef  typename MT::node_type                   NT;
  typedef  Tpetra::Map<LT,GT,NT>                    MAP;
  typedef  Tpetra::CrsGraph<LT,GT,NT>               GRAPH;
  typedef  Tpetra::Import<LT,GT,NT>                 IMPORT;
  typedef  Tpetra::Export<LT,GT,NT>                 EXPORT;
  typedef  Teuchos::Comm<int>                       COMM;
  

  //create all wrappers

  //Map Types
  typedef Teuchos::RCP<const MAP>  MAP_TYPE;
  //typedef LT*                       LT_ARRAY;
  //typedef GT*                       GT_ARRAY;
  //typedef ST*                       ST_ARRAY;

  typedef Teuchos::ArrayView<LT>             LT_ARRAY;
  typedef Teuchos::ArrayView<GT>             GT_ARRAY;
  typedef Teuchos::ArrayView<ST>             ST_ARRAY;


  typedef Teuchos::RCP<IMPORT>               IMPORT_TYPE;
  typedef Teuchos::RCP<EXPORT>               EXPORT_TYPE;



  static const LT LT_IDX_INVALID = Teuchos::OrdinalTraits<LT>::invalid();
  static const GT GT_IDX_INVALID = Teuchos::OrdinalTraits<GT>::invalid();


  inline
  static
  size_t MyPID(Matrix *A)
  {
    Teuchos::RCP<const Teuchos::Comm<int> > MatrixComm = A->getComm();
    return Teuchos::rank(*MatrixComm);
  }

  inline
  static
  COMM* NewSelfComm()
  {
#ifdef HAVE_SHYLUCORE_MPI
    return (new Teuchos::MpiComm<int>(MPI_COMM_SELF));
#else
    return (Teuchos::createSerialComm());
#endif
  }

  inline
  static
  COMM* GetComm(Matrix *A)
  {
    return A->getComm();
  }

  inline
  static
  COMM* GetComm(Vector *vec)
  {
    return (vec->getMap()->getComm());
  }

  inline
  static
  GT  NumGlobalRows(Matrix *A)
  {
    return A->getGlobalNumRows();
  }


  inline 
  static
  MAP* NewMap(GT NumGlobal, LT NumMyElements, GT* Elements,
              LT IndexBase, const COMM & comm)
  {
    return (new MAP(NumGlobal,Elements,IndexBase,comm));

  }


  inline
  static
  MAP_TYPE ColMap(Matrix *A)
  {
    return A->getColMap();
  }

  inline
  static
  MAP_TYPE RowMap(Matrix *A)
  {
    return A->getRowMap();
  }

  inline
  static
  MAP_TYPE DomainMap(Matrix *A)
  {
    return A->getDomainMap();
  }

  inline
  static
  MAP_TYPE RangeMap(Matrix &A)
  {
    return A.getRowMap();
  }

  inline
  static
  void ExtractMyRowView(Matrix *A, GT i, GT *nele, 
                        ST_ARRAY val, GT_ARRAY idx)
  {
    A->getLocalRowView(i,idx,val);
    *nele = idx.size();
  }

  inline
  static
  size_t  NumMyElements(MAP_TYPE A)
  {
    return A->getNodeNumElements;
  }

  inline
  static
  GT_ARRAY MyGlobalElements(MAP_TYPE A)
  {
    return A->getNodeElementList();
  }


  //Note that this will not work if GT or LT is unsigned
  inline
  static 
  GT  GID(MAP_TYPE A, LT i)
  {
    GT gid = A->getGlobalElement(i);
    if(gid == Teuchos::OrdinalTraits<GT>::invalid())
      {
        gid = -1;
      }
    return gid;
  }

  inline 
  static
  LT  LID(MAP_TYPE A, GT i)
  {
    LT lid = A->getLocalElement(i);
    if(lid == Teuchos::OrdinalTraits<LT>::invalid())
      {
        lid = -1;
      }
    return lid;
  }


  inline
  static
  void  SumAll(Matrix *A, GT *lvals, GT *gvals, GT n)
  {
    Teuchos::RCP<const Teuchos::Comm<int> > MatrixComm = A->getComm();
    Teuchos::reduceAll(*MatrixComm, Teuchos::REDUCE_SUM, n, lvals, gvals);
  }


  inline
  static 
  MT* NewMatrix(const MAP_TYPE &rowmap, const MAP_TYPE &colmap,
                GT *perrow)
  {
    //Might have issue with size_t
    GT numRows = rowmap.getGlobalNumElements();
    Teuchos::ArrayRCP<GT> temp = Teuchos::arcp(perrow, 0, numRows, false);
    return (new MT(rowmap, colmap, temp, Tpetra::StaticProfile)); 
  }

  
  inline 
  static
  MT* NewMatrix(const MAP_TYPE &rowmap, GT* perrow)
  {
     //Might have issue with size_t
    GT numRows = rowmap.getGlobalNumElements();
    Teuchos::ArrayRCP<GT> temp = Teuchos::arcp(perrow, 0, numRows, false);
    return (new MT(rowmap, temp, Tpetra::StaticProfile)); 

  }


  inline
  static
  void FillComplete(MT *A)
  {
    A->fillComplete();
  }
  
  inline
  static
  void FillComplete(MT *A, const MAP_TYPE &dmap, const MAP_TYPE &rmap)
  {
    A->fillComplete((dmap), (rmap));
  }

  inline
  static
  void ExtractView(VT *vec, ST** values, GT &lda)
  {
    //vec->ExtractView(values, &lda);
  }

  
  inline
  static
  int InsertGlobalValues(MT* A, const GT globalrow, const GT n,
                         ST* values, GT* idx)
  {
    //Note: ArrayView are unmannaged, so should not have to worry
    A->insertGlobalValues(globalrow, 
                               Teuchos::arrayView<const GT>(idx, n),
                               Teuchos::arrayView<const ST>(values,n)
                               );
    return 0;
  }

  inline
  static
  int ReplaceMyValues(MT* A, const LT localrow, const LT n,
                      ST* values, LT* idx)
  {
    A->replaceLocakValues(localrow,
                          Teuchos::arrayView<const LT>(idx, n),
                          Teuchos::arrayView<const ST>(values, n));
     return 0;
  }


  inline 
  static
  GRAPH* NewGraph(const MAP_TYPE &rowmap, GT* perrow)
  {
    GT numRows = rowmap.getGlobalNumElements();
    Teuchos::ArrayRCP<GT> temp = Teuchos::arcp(perrow, 0, numRows, false);
    return (new GRAPH(rowmap, temp, Tpetra::DynamicProfile)); 

  }

  inline
  static
  void FillComplete(GRAPH *graph)
  {
    graph->fillComplete();
  }

  inline
  static
  int InsertGlobalIndices(GRAPH *A,const GT grow,const GT n, GT* idx)
  {
    A->insterGlobalIndices(grow,
                           Teuchos::arrayView<const GT>(idx, n));
     return 0;
  }



  //Makes Vector
  inline
  static
  VT* NewVector(const MAP_TYPE &map, const size_t numVecs)
  {
    return (new Vector(map, numVecs));
  }

  //Makes View of Vector
  inline
  static
  VT* NewVectorView(VT *vec, size_t start, size_t numVecs)
  {
    
    std::cout << "ERROR Vector View for tpetra not defined" 
              << std::endl;
    return NULL;

  }

  inline
  static
  void VectorPutScalar(VT *vec, const ST &value)
  {
    vec->putScalar(value);
  }

  inline 
  static
  LT NumVectors(VT *vec)
  {
    return vec->getNumVectors();
  }

  inline
  static
  MAP* Map(VT *vec)
  {
    return (vec->getMap());
  }

  inline
  static
  void ReplaceMyValues(VT *vec, GT myrow, GT idx, ST value)
  {
    vec->replaceLocalValue(myrow, idx, value);
  }

  
  inline
  static
  void Import(VT *vec_target, VT &vec_source, IMPORT &importer)
  {
    vec_target->doImport(vec_source, importer, Tpetra::INSERT);
  }

  inline
  static
  IMPORT* NewImport(const MAP &TargetMap, const MAP &SourceMap)
  {
    return (new IMPORT(Teuchos::rcp(SourceMap,false),
                       Teuchos::rcp(TargetMap,false)));

  }
      
  inline
  static
  EXPORT* NewExport(const MAP &SourceMap, const MAP &TargetMap)
  {
    return (new EXPORT(Teuchos::rcp(SourceMap, false),
                       Teuchos::rcp(TargetMap, false)));
    
  }
      

#endif //end ifdef tpetra
};


//epetra_crsmatrix

//#ifdef HAVE_SHYLUCORE_EPETRA
template <>
struct ShyLUStackMap <Epetra_CrsMatrix,Epetra_MultiVector>
{
  typedef  Epetra_CrsMatrix                 MT;
  typedef  Epetra_MultiVector               VT;
  typedef  double                           ST;
  typedef  int                              LT; 
  typedef  int                              GT;
  typedef  int                              NT;
  typedef  Epetra_Map                      MAP;
  typedef  Epetra_CrsGraph                 GRAPH;
  typedef  Epetra_Import                   IMPORT;
  typedef  Epetra_Export                   EXPORT;
  typedef  Epetra_Comm                     COMM;

 /*
#ifdef HAVE_SHYLUCORE_MPI
  typedef  Epetra_MpiComm                  COMM;
#else
  typedef  Epetra_SerialComm               COMM;
#endif
  */
  //create all wrappers

  //Map Types
  typedef Teuchos::RCP<const MAP>            MAP_TYPE;
  typedef Teuchos::ArrayView<LT>             LT_ARRAY;
  typedef Teuchos::ArrayView<GT>             GT_ARRAY;
  typedef Teuchos::ArrayView<ST>             ST_ARRAY;

  typedef Teuchos::RCP<IMPORT>               IMPORT_TYPE;
  typedef Teuchos::RCP<EXPORT>               EXPORT_TYPE;

  static const LT LT_IDX_INVALID =  -1;
  static const GT GT_IDX_INVALID =  -1;

  inline
  static
  size_t MyPID(MT *A)
  {
    return A->Comm().MyPID();
  }

  inline
  static
  COMM* NewSelfComm()
  {
#ifdef HAVE_SHYLUCORE_MPI
    return (new Epetra_MpiComm(MPI_COMM_SELF));
#else
    return (new Epetra_SerialComm);
#endif //HAVE_SHYLUCORE_MPI 
  }

  inline
  static
  const COMM* GetComm(MT *A)
  {
    return &(A->Comm());
  }
  
  inline
  static
  const COMM* GetComm(VT *vec)
  {
    return &(vec->Comm());
  }

  inline
  static
  GT  NumGlobalRows(MT *A)
  {
    return A->NumGlobalRows();
  }


  inline
  static
  MAP* NewMap(GT NumGlobal, LT  NumMyElements, GT* Elements,
              GT IndexBase, const COMM &comm)
  {
    return (new MAP(NumGlobal, NumMyElements, Elements, IndexBase, comm));

  }

  inline
  static
  MAP_TYPE ColMap(MT *A)
  {
    return Teuchos::rcpFromRef(A->ColMap());
  }

  inline
  static
  MAP_TYPE RowMap(MT *A)
  {
    return Teuchos::rcpFromRef(A->RowMap());
  }

  inline
  static
  MAP_TYPE DomainMap(MT *A)
  {
    return Teuchos::rcpFromRef(A->DomainMap());
  }

  inline
  static
  MAP_TYPE RangeMap(MT *A)
  {
    return Teuchos::rcpFromRef(A->RowMap());
  }

  inline
  static
  void ExtractMyRowView(MT *A, GT i, GT *nele, ST_ARRAY &val, GT_ARRAY &idx)
  {
    ST *pval;
    GT *pidx;
    (void) A->ExtractMyRowView(i,*nele, pval ,pidx);
    val = Teuchos::arrayView(pval, *nele);
    idx = Teuchos::arrayView(pidx, *nele);
  }

  inline
  static
  size_t  NumMyElements(MAP_TYPE A)
  {
    return A->NumMyElements();
  }

  inline
  static
  GT_ARRAY MyGlobalElements(MAP_TYPE A)
  {
    return Teuchos::arrayView(A->MyGlobalElements(), NumMyElements(A));
  }

  //Note that this will not work if GT or LT is unsigned
  inline
  static 
  GT  GID(MAP_TYPE A, LT i)
  {
    return A->GID(i);
  }

  inline 
  static
  LT  LID(MAP_TYPE A, GT i)
  {
    return A->LID(i);
  }


  inline
  static
  void  SumAll(MT *A, GT *lvals, GT *gvals, GT n)
  {
    A->Comm().SumAll(lvals, gvals, n);
  }

  inline
  static 
  MT* NewMatrix(const MAP_TYPE &rowmap, const MAP_TYPE &colmap,
                GT *perrow)
  {
    return (new MT(Copy, *(rowmap), *(colmap), perrow, true));
  }

  inline
  static
  MT* NewMatrix(const MAP_TYPE &rowmap, GT* perrow)
  {
    return (new MT(Copy, *(rowmap), perrow, true));
  }
  
  inline
  static
  void FillComplete(MT *A)
  {
    A->FillComplete();
  }
  
  inline
  static
  void FillComplete(MT *A, const MAP_TYPE &dmap, const MAP_TYPE &rmap)
  {
    A->FillComplete(*(dmap), *(rmap));
  }
  
  inline
  static
  int InsertGlobalValues(MT* A, const GT globalrow, const GT n,
                         ST* values, GT* idx)
  {
    return A->InsertGlobalValues(globalrow, n, values, idx);
  }

  inline
  static
  int ReplaceMyValues(MT* A, const LT localrow, const LT n,
                      ST* values, LT* idx)
  {
    return A->ReplaceMyValues(localrow, n, values, idx);
  }

  inline
  static
  GRAPH* NewGraph(const MAP_TYPE &rowmap, GT* perrow)
  {
    return (new GRAPH(Copy, *(rowmap), perrow, false));
  }

  inline
  static
  void FillComplete(GRAPH *graph)
  {
    graph->FillComplete();
  }


  inline
  static
  int InsertGlobalIndices(GRAPH *A,const GT grow,const GT n, GT* idx)
  {
    return A->InsertGlobalIndices(grow, n, idx);
  }

   //Makes Vector
  inline
  static
  VT* NewVector(const MAP_TYPE &map, const size_t numVecs)
  {
    return (new VT(*(map), ((int)numVecs)));
  }

  //Makes View of Vector
  inline
  static
  VT* NewVectorView(VT *vec, size_t start, size_t numVecs)
  {
    return (new VT(View, *vec, start, numVecs));
  }

  inline
  static
  void VectorPutScalar(VT *vec, const ST &value)
  {
    vec->PutScalar(value);
  }

  inline 
  static
  LT NumVectors(VT *vec)
  {
    return vec->NumVectors();
  }

  inline
  static
  MAP* Map(VT *vec)
  {
    return vec->Map();
  }

  inline
  static
  void ReplaceMyValues(VT *vec, GT myrow, GT idx, ST value)
  {
    vec->ReplaceMyValue(myrow, idx, value);
  }
  
  inline
  static
  void ExtractView(VT *vec, ST** values, GT &lda)
  {
    vec->ExtractView(values, &lda);
  }

  inline
  static
  void Import(VT *vec_target, VT &vec_source, IMPORT &importer)
  {
    vec_target->Import(vec_source, importer, Insert);
  }

  inline
  static
  IMPORT* NewImport(const MAP &TargetMap, const MAP &SourceMap)
  {
    return (new IMPORT(TargetMap,SourceMap));
  } 

  inline
  static
  EXPORT* NewExport(const MAP &SourceMap, const MAP &TargetMap)
  {
    return (new EXPORT(SourceMap, TargetMap));
  }


};
//#endif // HAVE_SHYLU_EPETRA


#endif //ends SHYLU_TYPEMPP
