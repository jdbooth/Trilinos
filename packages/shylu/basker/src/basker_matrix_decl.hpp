#ifndef BASKER_MATRIX_DECL_HPP
#define BASKER_MATRIX_DECL_HPP

/*Basker Includes*/
//#include "basker_decl.hpp"
#include "basker_types.hpp"

/*System Includes*/
#include <limits>
#include <string>

#ifdef BASKER_KOKKOS
#include <Kokkos_Core.hpp>
#else
#include <omp.h>
#endif

using std::string;

namespace BaskerNS
{
  template <class Int, class Entry, class Exe_Space>
  class BaskerMatrix
  {
 
  public:
    
    //Constructors and deconstructors
    BASKER_INLINE
    BaskerMatrix();
    BASKER_INLINE
    BaskerMatrix(string _label);
    BASKER_INLINE
    BaskerMatrix(Int _m, Int _n, Int _nnz, 
                 Int *col_ptr, Int *row_idx, Entry *val);
    BASKER_INLINE
    BaskerMatrix(string _label, Int _m, Int _n, Int _nnz, 
                 Int *col_ptr, Int *row_idx, Entry *val);
    BASKER_INLINE
    ~BaskerMatrix();

    /*
    BASKER_INLINE
    BaskerMatrix<Int,Entry,Exe_Space>& operator= (const BaskerMatrix<Int,Entry,Exe_Space>&);
    */


    //init_matrix (want to change these to malloc_matrix)
    BASKER_INLINE
    void init_matrix(string _label, Int _m, Int _n, Int _nnz);
    BASKER_INLINE
    void init_matrix(string _label, Int _m, Int _n, Int _nnz,
                    Int *_col_ptr, Int *_row_idx, Entry *_val);
    BASKER_INLINE
    void init_matrix(string _label, Int _sr, Int _m, 
                    Int _sc, Int _n, Int _nnz);
    BASKER_INLINE
    int copy_values(Int _sr, Int _m, Int _sc, Int _n, Int _nnz,
		    Int *_col_ptr, Int *_row_idx, Entry *_val);
    BASKER_INLINE
    int copy_values(Int _m, Int _n, Int _nnz,
		    Int *_col_ptr, Int *_row_idx, Entry *_val);



    BASKER_INLINE
    void init_col();
    BASKER_INLINE
    void clean_col();
    BASKER_INLINE
    void convert2D(BASKER_MATRIX &M, 
		   BASKER_BOOL alloc = BASKER_TRUE);

    
    //just set shape, do not init
    void set_shape(Int _sr, Int _m, 
		  Int _sc, Int _n);


    BASKER_INLINE
    int fill();

    BASKER_BOOL v_fill;

    //malloc perm
    //BASKER_INLINE
    //int malloc_perm();
    BASKER_INLINE
    void malloc_perm(Int n);
    BASKER_INLINE
    void init_perm();

    //malloc union_bit
    BASKER_INLINE
    void malloc_union_bit();
    BASKER_INLINE
    void init_union_bit();
    BASKER_INLINE
    void init_union_bit(Int kid);

    //helper functions
    void copy_vec(Int* ptr, Int size, INT_1DARRAY a);
    void copy_vec(Entry *ptr, Int size,  ENTRY_1DARRAY a);
    BASKER_INLINE
    void init_vectors(Int _m, Int _n, Int _nnz);


    //information
    BASKER_INLINE
    void info();
    BASKER_INLINE
    void level_info();
    BASKER_INLINE
    void print();


    string label;

    Int srow, scol;
    Int erow, ecol;
    Int ncol, nrow, nnz;
    
    INT_1DARRAY   col_ptr;
    INT_1DARRAY   row_idx;
    ENTRY_1DARRAY val;
    INT_1DARRAY   lpinv;

    BOOL_1DARRAY union_bit;
   
    

    #ifdef BASKER_INC_LVL
    INT_1DARRAY   inc_lvl;
    #endif

  
    #ifdef BASKER_2DL
    ENTRY_1DARRAY ews;
    INT_1DARRAY   iws;
    Int           iws_size;
    Int           ews_size;
    Int           iws_mult;
    Int           ews_mult;
    Int           p_size;
    #endif
    

    Entry tpivot;

    //Remove..... will not be used in future ver
    static const Int max_idx = (Int) -1;
    //static const Int max_idx = std::numeric_limits<Int>::max();
    // static Int max_idx = std::numeric_limits<Int>::max();
    
    //Used for end
    INT_1DARRAY pend;
    void init_pend();
    

  };//end class BaskerMatrix

}//End namespace BaskerNS

#endif //End ifndef basker_matrix_decl_hpp
