// @HEADER
// ***********************************************************************
//
//           Panzer: A partial differential equation assembly
//       engine for strongly coupled complex multiphysics systems
//                 Copyright (2011) Sandia Corporation
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
// Questions? Contact Roger P. Pawlowski (rppawlo@sandia.gov) and
// Eric C. Cyr (eccyr@sandia.gov)
// ***********************************************************************
// @HEADER

#ifndef PANZER_WORKSET_BUILDER_IMPL_HPP
#define PANZER_WORKSET_BUILDER_IMPL_HPP

#include <iostream>
#include <vector>
#include <map>
#include <algorithm>
#include "Panzer_Workset.hpp"
#include "Panzer_CellData.hpp"
#include "Panzer_BC.hpp"
#include "Panzer_PhysicsBlock.hpp"
#include "Panzer_Shards_Utilities.hpp"
#include "Panzer_CommonArrayFactories.hpp"

#include "Phalanx_DataLayout_MDALayout.hpp"

// Intrepid2
#include "Shards_CellTopology.hpp"
#include "Intrepid2_DefaultCubatureFactory.hpp"
#include "Intrepid2_CellTools.hpp"
#include "Intrepid2_FunctionSpaceTools.hpp"
#include "Intrepid2_Basis.hpp"

template<typename ArrayT>
Teuchos::RCP< std::vector<panzer::Workset> > 
panzer::buildWorksets(const panzer::PhysicsBlock& pb,
		      const std::vector<std::size_t>& local_cell_ids,
		      const ArrayT& vertex_coordinates)
{
  using Teuchos::RCP;
  using Teuchos::rcp;

  WorksetNeeds needs;
  needs.cellData = pb.cellData();

  const std::map<int,RCP<panzer::IntegrationRule> >& int_rules = pb.getIntegrationRules();
  for(std::map<int,RCP<panzer::IntegrationRule> >::const_iterator ir_itr = int_rules.begin();
      ir_itr != int_rules.end(); ++ir_itr)
    needs.int_rules.push_back(ir_itr->second);
  
  const std::map<std::string,Teuchos::RCP<panzer::PureBasis> >& bases= pb.getBases();
  for(std::map<std::string,Teuchos::RCP<panzer::PureBasis> >::const_iterator b_itr = bases.begin();
      b_itr != bases.end(); ++b_itr)
    needs.bases.push_back(b_itr->second);
 
  return buildWorksets(needs,pb.elementBlockID(),local_cell_ids,vertex_coordinates);
}

template<typename ArrayT>
Teuchos::RCP< std::vector<panzer::Workset> > 
panzer::buildWorksets(const WorksetNeeds & needs,
                      const std::string & elementBlock,
		      const std::vector<std::size_t>& local_cell_ids,
		      const ArrayT& vertex_coordinates)
{
  using std::vector;
  using std::string;
  using Teuchos::RCP;
  using Teuchos::rcp;

  panzer::MDFieldArrayFactory mdArrayFactory("",true);

  std::size_t total_num_cells = local_cell_ids.size();

  std::size_t workset_size = needs.cellData.numCells();

  Teuchos::RCP< std::vector<panzer::Workset> > worksets_ptr = 
    Teuchos::rcp(new std::vector<panzer::Workset>);
  std::vector<panzer::Workset>& worksets = *worksets_ptr;
   
  // special case for 0 elements!
  if(total_num_cells==0) {

     // Setup integration rules and basis
     RCP<vector<int> > ir_degrees = rcp(new vector<int>(0));
     RCP<vector<string> > basis_names = rcp(new vector<string>(0));
      
     worksets.resize(1);
     std::vector<panzer::Workset>::iterator i = worksets.begin();
     i->num_cells = 0;
     i->block_id = elementBlock;
     i->ir_degrees = ir_degrees;
     i->basis_names = basis_names;

     for (std::size_t j=0;j<needs.int_rules.size();j++) {
         
       RCP<panzer::IntegrationValues2<double> > iv2 = 
	 rcp(new panzer::IntegrationValues2<double>("",true));
       iv2->setupArrays(needs.int_rules[j]);

       ir_degrees->push_back(needs.int_rules[j]->cubature_degree);
       i->int_rules.push_back(iv2);
     }

     // Need to create all combinations of basis/ir pairings 
     for (std::size_t j=0;j<needs.int_rules.size();j++) {
       for (std::size_t b=0;b<needs.bases.size();b++) {
	 RCP<panzer::BasisIRLayout> b_layout 
             = rcp(new panzer::BasisIRLayout(needs.bases[b],*needs.int_rules[j]));
	 
	 RCP<panzer::BasisValues2<double> > bv2 
             = rcp(new panzer::BasisValues2<double>("",true,true));
	 bv2->setupArrays(b_layout);
	 i->bases.push_back(bv2);

	 basis_names->push_back(b_layout->name());
       }

     }

     return worksets_ptr;
  } // end special case

  {
    std::size_t num_worksets = total_num_cells / workset_size;
    bool last_set_is_full = true;
    std::size_t last_workset_size = total_num_cells % workset_size;
    if (last_workset_size != 0) {
      num_worksets += 1;
      last_set_is_full = false;
    }    

    worksets.resize(num_worksets);
    std::vector<panzer::Workset>::iterator i;
    for (i = worksets.begin(); i != worksets.end(); ++i)
      i->num_cells = workset_size;
	 
    if (!last_set_is_full) {
      worksets.back().num_cells = last_workset_size;
    }
  }

  // assign workset cell local ids
  std::vector<std::size_t>::const_iterator local_begin = local_cell_ids.begin();
  for (std::vector<panzer::Workset>::iterator wkst = worksets.begin(); wkst != worksets.end(); ++wkst) {
    std::vector<std::size_t>::const_iterator begin_iter = local_begin;
    std::vector<std::size_t>::const_iterator end_iter = begin_iter + wkst->num_cells;
    local_begin = end_iter;
    wkst->cell_local_ids.assign(begin_iter,end_iter);

    Kokkos::View<int*,PHX::Device> cell_local_ids_k = Kokkos::View<int*,PHX::Device>("Workset:cell_local_ids",wkst->cell_local_ids.size());
    for(std::size_t i=0;i<wkst->cell_local_ids.size();i++)
      cell_local_ids_k(i) = wkst->cell_local_ids[i];
    wkst->cell_local_ids_k = cell_local_ids_k;
    
    wkst->cell_vertex_coordinates = mdArrayFactory.buildStaticArray<double,Cell,NODE,Dim>("cvc",workset_size,
					 vertex_coordinates.dimension(1),
					 vertex_coordinates.dimension(2));
    wkst->block_id = elementBlock;
    wkst->subcell_dim = needs.cellData.baseCellDimension();
    wkst->subcell_index = 0;
  }
  
  TEUCHOS_ASSERT(local_begin == local_cell_ids.end());

  // Copy cell vertex coordinates into local workset arrays
  std::size_t offset = 0;
  for (std::vector<panzer::Workset>::iterator wkst = worksets.begin(); wkst != worksets.end(); ++wkst) {
    for (std::size_t cell = 0; cell < wkst->num_cells; ++cell)
      for (std::size_t vertex = 0; vertex < Teuchos::as<std::size_t>(vertex_coordinates.dimension(1)); ++ vertex)
	for (std::size_t dim = 0; dim < Teuchos::as<std::size_t>(vertex_coordinates.dimension(2)); ++ dim) {
	  //wkst->cell_vertex_coordinates(cell,vertex,dim) = vertex_coordinates(cell + offset,vertex,dim);
	  wkst->cell_vertex_coordinates(cell,vertex,dim) = vertex_coordinates(cell + offset,vertex,dim);
        }

    offset += wkst->num_cells;
  }

  TEUCHOS_ASSERT(offset == Teuchos::as<std::size_t>(vertex_coordinates.dimension(0)));
  
  // Set ir and basis arrayskset
  RCP<vector<int> > ir_degrees = rcp(new vector<int>(0));
  RCP<vector<string> > basis_names = rcp(new vector<string>(0));
  for (std::vector<panzer::Workset>::iterator wkst = worksets.begin(); wkst != worksets.end(); ++wkst) {
    wkst->ir_degrees = ir_degrees;
    wkst->basis_names = basis_names;
  }

  // setup the integration rules and bases
  for(std::vector<panzer::Workset>::iterator wkst = worksets.begin(); wkst != worksets.end(); ++wkst)
    populateValueArrays(wkst->num_cells,false,needs,*wkst);

  return worksets_ptr;
}

// ****************************************************************
// ****************************************************************

template<typename ArrayT>
Teuchos::RCP<std::map<unsigned,panzer::Workset> >
panzer::buildBCWorkset(const panzer::PhysicsBlock & pb,
		       const std::vector<std::size_t>& local_cell_ids,
		       const std::vector<std::size_t>& local_side_ids,
		       const ArrayT& vertex_coordinates)
{
  using Teuchos::RCP;
  using Teuchos::rcp;

  WorksetNeeds needs;
  needs.cellData = pb.cellData();

  const std::map<int,RCP<panzer::IntegrationRule> >& int_rules = pb.getIntegrationRules();
  for(std::map<int,RCP<panzer::IntegrationRule> >::const_iterator ir_itr = int_rules.begin();
      ir_itr != int_rules.end(); ++ir_itr)
    needs.int_rules.push_back(ir_itr->second);
  
 const std::map<std::string,Teuchos::RCP<panzer::PureBasis> >& bases= pb.getBases();
 for(std::map<std::string,Teuchos::RCP<panzer::PureBasis> >::const_iterator b_itr = bases.begin();
     b_itr != bases.end(); ++b_itr)
   needs.bases.push_back(b_itr->second);
 
 return buildBCWorkset(needs,pb.elementBlockID(),local_cell_ids,local_side_ids,vertex_coordinates);
}

template<typename ArrayT>
Teuchos::RCP<std::map<unsigned,panzer::Workset> >
panzer::buildBCWorkset(const WorksetNeeds & needs,
                       const std::string & elementBlock,
                       const std::vector<std::size_t>& local_cell_ids,
                       const std::vector<std::size_t>& local_side_ids,
                       const ArrayT& vertex_coordinates,
                       const bool populate_value_arrays)
{
  using Teuchos::RCP;
  using Teuchos::rcp;

  panzer::MDFieldArrayFactory mdArrayFactory("",true);

  // key is local face index, value is workset with all elements
  // for that local face
  Teuchos::RCP<std::map<unsigned,panzer::Workset> > worksets_ptr = 
    Teuchos::rcp(new std::map<unsigned,panzer::Workset>);

  // All elements of boundary condition should go into one workset.
  // However due to design of Intrepid2 (requires same basis for all
  // cells), we have to separate the workset based on the local side
  // index.  Each workset for a boundary condition is associated with
  // a local side for the element
  
  TEUCHOS_ASSERT(local_side_ids.size() == local_cell_ids.size());
  TEUCHOS_ASSERT(local_side_ids.size() == static_cast<std::size_t>(vertex_coordinates.dimension(0)));

  // key is local face index, value is a pair of cell index and vector of element local ids
  std::map<unsigned,std::vector<std::pair<std::size_t,std::size_t> > > element_list;
  for (std::size_t cell=0; cell < local_cell_ids.size(); ++cell)
    element_list[local_side_ids[cell]].push_back(std::make_pair(cell,local_cell_ids[cell])); 

  std::map<unsigned,panzer::Workset>& worksets = *worksets_ptr;

  // create worksets 
  std::map<unsigned,std::vector<std::pair<std::size_t,std::size_t> > >::const_iterator side;
  for (side = element_list.begin(); side != element_list.end(); ++side) {

    std::vector<std::size_t>& cell_local_ids = worksets[side->first].cell_local_ids;

    worksets[side->first].cell_vertex_coordinates = mdArrayFactory.buildStaticArray<double,Cell,NODE,Dim>("cvc",
                                                          side->second.size(),
                                                          vertex_coordinates.dimension(1),
                                                          vertex_coordinates.dimension(2));
    Workset::CellCoordArray coords = worksets[side->first].cell_vertex_coordinates;

    for (std::size_t cell = 0; cell < side->second.size(); ++cell) {
      cell_local_ids.push_back(side->second[cell].second);

      for (std::size_t vertex = 0; vertex < Teuchos::as<std::size_t>(vertex_coordinates.dimension(1)); ++ vertex)
	for (std::size_t dim = 0; dim < Teuchos::as<std::size_t>(vertex_coordinates.dimension(2)); ++ dim) {
	  coords(cell,vertex,dim) = vertex_coordinates(side->second[cell].first,vertex,dim);
        }
    }

    Kokkos::View<int*,PHX::Device> cell_local_ids_k = Kokkos::View<int*,PHX::Device>("Workset:cell_local_ids",worksets[side->first].cell_local_ids.size());
    for(std::size_t i=0;i<worksets[side->first].cell_local_ids.size();i++)
      cell_local_ids_k(i) = worksets[side->first].cell_local_ids[i];
    worksets[side->first].cell_local_ids_k = cell_local_ids_k;

    worksets[side->first].num_cells = worksets[side->first].cell_local_ids.size();
    worksets[side->first].block_id = elementBlock;
    worksets[side->first].subcell_dim = needs.cellData.baseCellDimension() - 1;
    worksets[side->first].subcell_index = side->first;
  }

  if (populate_value_arrays) {
    // setup the integration rules and bases
    for (std::map<unsigned,panzer::Workset>::iterator wkst = worksets.begin();
         wkst != worksets.end(); ++wkst) {

      populateValueArrays(wkst->second.num_cells,true,needs,wkst->second); // populate "side" values
    }
  }

  return worksets_ptr;
}

// ****************************************************************
// ****************************************************************

template<typename ArrayT>
Teuchos::RCP< std::vector<panzer::Workset> > 
panzer::buildEdgeWorksets(const panzer::PhysicsBlock & pb_a,
	  	          const std::vector<std::size_t>& local_cell_ids_a,
		          const std::vector<std::size_t>& local_side_ids_a,
		          const ArrayT& vertex_coordinates_a,
                          const panzer::PhysicsBlock & pb_b,
		          const std::vector<std::size_t>& local_cell_ids_b,
		          const std::vector<std::size_t>& local_side_ids_b,
		          const ArrayT& vertex_coordinates_b)
{
  using Teuchos::RCP;
  using Teuchos::rcp;

  WorksetNeeds needs_a;
  needs_a.cellData = pb_a.cellData();

  const std::map<int,RCP<panzer::IntegrationRule> >& int_rules_a  = pb_a.getIntegrationRules();
  for(std::map<int,RCP<panzer::IntegrationRule> >::const_iterator ir_itr = int_rules_a.begin();
      ir_itr != int_rules_a.end(); ++ir_itr)
    needs_a.int_rules.push_back(ir_itr->second);
  
  const std::map<std::string,Teuchos::RCP<panzer::PureBasis> >& bases_a = pb_a.getBases();
  for(std::map<std::string,Teuchos::RCP<panzer::PureBasis> >::const_iterator b_itr = bases_a.begin();
      b_itr != bases_a.end(); ++b_itr)
    needs_a.bases.push_back(b_itr->second);

  WorksetNeeds needs_b;
  needs_b.cellData = pb_b.cellData();

  const std::map<int,RCP<panzer::IntegrationRule> >& int_rules_b  = pb_b.getIntegrationRules();
  for(std::map<int,RCP<panzer::IntegrationRule> >::const_iterator ir_itr = int_rules_b.begin();
      ir_itr != int_rules_b.end(); ++ir_itr)
    needs_b.int_rules.push_back(ir_itr->second);
  
  const std::map<std::string,Teuchos::RCP<panzer::PureBasis> >& bases_b = pb_b.getBases();
  for(std::map<std::string,Teuchos::RCP<panzer::PureBasis> >::const_iterator b_itr = bases_b.begin();
      b_itr != bases_b.end(); ++b_itr)
    needs_b.bases.push_back(b_itr->second);

  return buildEdgeWorksets(needs_a,pb_a.elementBlockID(),local_cell_ids_a,local_side_ids_a,vertex_coordinates_a,
                           needs_b,pb_b.elementBlockID(),local_cell_ids_b,local_side_ids_b,vertex_coordinates_b);
/*
  using std::vector;
  using std::string;
  using Teuchos::RCP;
  using Teuchos::rcp;

  panzer::MDFieldArrayFactory mdArrayFactory("",true);

  std::size_t total_num_cells_a = local_cell_ids_a.size();
  std::size_t total_num_cells_b = local_cell_ids_b.size();

  TEUCHOS_ASSERT(total_num_cells_a==total_num_cells_b);
  TEUCHOS_ASSERT(local_side_ids_a.size() == local_cell_ids_a.size());
  TEUCHOS_ASSERT(local_side_ids_a.size() == static_cast<std::size_t>(vertex_coordinates_a.dimension(0)));
  TEUCHOS_ASSERT(local_side_ids_b.size() == local_cell_ids_b.size());
  TEUCHOS_ASSERT(local_side_ids_b.size() == static_cast<std::size_t>(vertex_coordinates_b.dimension(0)));

  std::size_t total_num_cells = total_num_cells_a;

  std::size_t workset_size = pb_a.cellData().numCells();

  Teuchos::RCP< std::vector<panzer::Workset> > worksets_ptr = 
    Teuchos::rcp(new std::vector<panzer::Workset>);
  std::vector<panzer::Workset>& worksets = *worksets_ptr;
   
  // special case for 0 elements!
  if(total_num_cells==0) {

     // Setup integration rules and basis
     RCP<vector<int> > ir_degrees = rcp(new vector<int>(0));
     RCP<vector<string> > basis_names = rcp(new vector<string>(0));
      
     worksets.resize(1);
     std::vector<panzer::Workset>::iterator i = worksets.begin();

     i->details(0)->block_id = pb_a.elementBlockID();
     i->other = Teuchos::rcp(new panzer::WorksetDetails);
     i->details(1).block_id = pb_b.elementBlockID();

     i->num_cells = 0;
     i->ir_degrees = ir_degrees;
     i->basis_names = basis_names;

     const std::map<int,RCP<panzer::IntegrationRule> >& int_rules = pb_a.getIntegrationRules();
     
     for (std::map<int,RCP<panzer::IntegrationRule> >::const_iterator ir_itr = int_rules.begin();
	  ir_itr != int_rules.end(); ++ir_itr) {
         
      RCP<panzer::IntegrationValues2<double> > iv2 = 
         rcp(new panzer::IntegrationValues2<double>("",true));
       iv2->setupArrays(ir_itr->second);

       ir_degrees->push_back(ir_itr->first);
       // i->int_rules.push_back(iv);
       i->int_rules.push_back(iv2);
     }

     const std::map<std::string,Teuchos::RCP<panzer::PureBasis> >& bases = pb_a.getBases();

     // Need to create all combinations of basis/ir pairings 
     for (std::map<int,RCP<panzer::IntegrationRule> >::const_iterator ir_itr = int_rules.begin();
	  ir_itr != int_rules.end(); ++ir_itr) {

       for (std::map<std::string,Teuchos::RCP<panzer::PureBasis> >::const_iterator b_itr = bases.begin();
	    b_itr != bases.end(); ++b_itr) {

	 RCP<panzer::BasisIRLayout> b_layout = rcp(new panzer::BasisIRLayout(b_itr->second,*ir_itr->second));
	 
	 RCP<panzer::BasisValues2<double> > bv2 = 
	   rcp(new panzer::BasisValues2<double>("",true,true));
	 bv2->setupArrays(b_layout);
	 i->bases.push_back(bv2);

	 basis_names->push_back(b_layout->name());
       }

     }

     return worksets_ptr;
  } // end special case

  // This collects all the elements that share the same sub cell pairs, this makes it easier to
  // build the required worksets
  // key is the pair of local face indices, value is a vector of cell indices that satisfy this pair
  std::map<std::pair<unsigned,unsigned>,std::vector<std::size_t> > element_list;
  for (std::size_t cell=0; cell < local_cell_ids_a.size(); ++cell)
    element_list[std::make_pair(local_side_ids_a[cell],local_side_ids_b[cell])].push_back(cell);

  // this is the lone iterator that will be used to loop over the element edge list
  std::map<std::pair<unsigned,unsigned>,std::vector<std::size_t> >::const_iterator edge;

  // figure out how many worksets will be needed, resize workset vector accordingly
  std::size_t num_worksets = 0;
  for(edge=element_list.begin(); edge!=element_list.end();++edge) {
    std::size_t num_worksets_for_edge = edge->second.size() / workset_size;
    std::size_t last_workset_size = edge->second.size() % workset_size;
    if(last_workset_size!=0)
      num_worksets_for_edge += 1;

    num_worksets += num_worksets_for_edge;
  }
  worksets.resize(num_worksets);

  // fill the worksets
  std::vector<Workset>::iterator current_workset = worksets.begin();
  for(edge=element_list.begin(); edge!=element_list.end();++edge) {
    // loop over each workset
    const std::vector<std::size_t> & cell_indices = edge->second;
    
    current_workset = buildEdgeWorksets(cell_indices,
                                       pb_a,local_cell_ids_a,local_side_ids_a,vertex_coordinates_a,
                                       pb_b,local_cell_ids_b,local_side_ids_b,vertex_coordinates_b,
                                       current_workset);
  }

  // sanity check
  TEUCHOS_ASSERT(current_workset==worksets.end());

  return worksets_ptr;
*/
}

template<typename ArrayT>
Teuchos::RCP<std::vector<panzer::Workset> > 
panzer::buildEdgeWorksets(const WorksetNeeds & needs_a,
                   const std::string & eblock_a,
	 	   const std::vector<std::size_t>& local_cell_ids_a,
		   const std::vector<std::size_t>& local_side_ids_a,
		   const ArrayT& vertex_coordinates_a,
                   const WorksetNeeds & needs_b,
                   const std::string & eblock_b,
		   const std::vector<std::size_t>& local_cell_ids_b,
		   const std::vector<std::size_t>& local_side_ids_b,
		   const ArrayT& vertex_coordinates_b)
{
  using Teuchos::RCP;
  using Teuchos::rcp;

  panzer::MDFieldArrayFactory mdArrayFactory("",true);

  std::size_t total_num_cells_a = local_cell_ids_a.size();
  std::size_t total_num_cells_b = local_cell_ids_b.size();

  TEUCHOS_ASSERT(total_num_cells_a==total_num_cells_b);
  TEUCHOS_ASSERT(local_side_ids_a.size() == local_cell_ids_a.size());
  TEUCHOS_ASSERT(local_side_ids_a.size() == static_cast<std::size_t>(vertex_coordinates_a.dimension(0)));
  TEUCHOS_ASSERT(local_side_ids_b.size() == local_cell_ids_b.size());
  TEUCHOS_ASSERT(local_side_ids_b.size() == static_cast<std::size_t>(vertex_coordinates_b.dimension(0)));

  std::size_t total_num_cells = total_num_cells_a;

  std::size_t workset_size = needs_a.cellData.numCells();

  Teuchos::RCP< std::vector<panzer::Workset> > worksets_ptr = 
    Teuchos::rcp(new std::vector<panzer::Workset>);
  std::vector<panzer::Workset>& worksets = *worksets_ptr;
   
  // special case for 0 elements!
  if(total_num_cells==0) {

     // Setup integration rules and basis
     RCP<std::vector<int> > ir_degrees = rcp(new std::vector<int>(0));
     RCP<std::vector<std::string> > basis_names = rcp(new std::vector<std::string>(0));
      
     worksets.resize(1);
     std::vector<panzer::Workset>::iterator i = worksets.begin();

     i->details(0).block_id = eblock_a;
     i->other = Teuchos::rcp(new panzer::WorksetDetails);
     i->details(1).block_id = eblock_b;

     i->num_cells = 0;
     i->ir_degrees = ir_degrees;
     i->basis_names = basis_names;

     for(std::size_t j=0;j<needs_a.int_rules.size();j++) {
         
      RCP<panzer::IntegrationValues2<double> > iv2 = 
         rcp(new panzer::IntegrationValues2<double>("",true));
       iv2->setupArrays(needs_a.int_rules[j]);

       ir_degrees->push_back(needs_a.int_rules[j]->cubature_degree);
       i->int_rules.push_back(iv2);
     }

     // Need to create all combinations of basis/ir pairings 
     for(std::size_t j=0;j<needs_a.int_rules.size();j++) {

        for(std::size_t b=0;b<needs_a.bases.size();b++) {

	 RCP<panzer::BasisIRLayout> b_layout = rcp(new panzer::BasisIRLayout(needs_a.bases[b],*needs_a.int_rules[j]));
	 
	 RCP<panzer::BasisValues2<double> > bv2 = 
	   rcp(new panzer::BasisValues2<double>("",true,true));
	 bv2->setupArrays(b_layout);
	 i->bases.push_back(bv2);

	 basis_names->push_back(b_layout->name());
       }

     }

     return worksets_ptr;
  } // end special case

  // This collects all the elements that share the same sub cell pairs, this makes it easier to
  // build the required worksets
  // key is the pair of local face indices, value is a vector of cell indices that satisfy this pair
  std::map<std::pair<unsigned,unsigned>,std::vector<std::size_t> > element_list;
  for (std::size_t cell=0; cell < local_cell_ids_a.size(); ++cell)
    element_list[std::make_pair(local_side_ids_a[cell],local_side_ids_b[cell])].push_back(cell);

  // this is the lone iterator that will be used to loop over the element edge list
  std::map<std::pair<unsigned,unsigned>,std::vector<std::size_t> >::const_iterator edge;

  // figure out how many worksets will be needed, resize workset vector accordingly
  std::size_t num_worksets = 0;
  for(edge=element_list.begin(); edge!=element_list.end();++edge) {
    std::size_t num_worksets_for_edge = edge->second.size() / workset_size;
    std::size_t last_workset_size = edge->second.size() % workset_size;
    if(last_workset_size!=0)
      num_worksets_for_edge += 1;

    num_worksets += num_worksets_for_edge;
  }
  worksets.resize(num_worksets);

  // fill the worksets
  std::vector<Workset>::iterator current_workset = worksets.begin();
  for(edge=element_list.begin(); edge!=element_list.end();++edge) {
    // loop over each workset
    const std::vector<std::size_t> & cell_indices = edge->second;
    
    current_workset = buildEdgeWorksets(cell_indices,
                                       needs_a,eblock_a,local_cell_ids_a,local_side_ids_a,vertex_coordinates_a,
                                       needs_b,eblock_b,local_cell_ids_b,local_side_ids_b,vertex_coordinates_b,
                                       current_workset);
  }

  // sanity check
  TEUCHOS_ASSERT(current_workset==worksets.end());

  return worksets_ptr;
}

template<typename ArrayT>
std::vector<panzer::Workset>::iterator
panzer::buildEdgeWorksets(const std::vector<std::size_t> & cell_indices,
                          const WorksetNeeds & needs_a,
                          const std::string & eblock_a,
	 	          const std::vector<std::size_t>& local_cell_ids_a,
		          const std::vector<std::size_t>& local_side_ids_a,
		          const ArrayT& vertex_coordinates_a,
                          const WorksetNeeds & needs_b,
                          const std::string & eblock_b,
	      	          const std::vector<std::size_t>& local_cell_ids_b,
		          const std::vector<std::size_t>& local_side_ids_b,
		          const ArrayT& vertex_coordinates_b,
                          std::vector<Workset>::iterator beg)
{
  panzer::MDFieldArrayFactory mdArrayFactory("",true);

  std::vector<Workset>::iterator wkst = beg;
 
  std::size_t current_cell_index = 0;
  while (current_cell_index<cell_indices.size()) {
    std::size_t workset_size = needs_a.cellData.numCells();

    // allocate workset details (details(0) is already associated with the
    // workset object itself)
    wkst->other = Teuchos::rcp(new panzer::WorksetDetails);

    wkst->subcell_dim = needs_a.cellData.baseCellDimension()-1;

    wkst->details(0).subcell_index = local_side_ids_a[cell_indices[current_cell_index]];
    wkst->details(0).block_id = eblock_a;
    wkst->details(0).cell_vertex_coordinates = mdArrayFactory.buildStaticArray<double,Cell,NODE,Dim>("cvc",workset_size,
					 vertex_coordinates_a.dimension(1),
					 vertex_coordinates_a.dimension(2));

    wkst->details(1).subcell_index = local_side_ids_b[cell_indices[current_cell_index]];
    wkst->details(1).block_id = eblock_b; 
    wkst->details(1).cell_vertex_coordinates = mdArrayFactory.buildStaticArray<double,Cell,NODE,Dim>("cvc",workset_size,
					 vertex_coordinates_a.dimension(1),
					 vertex_coordinates_a.dimension(2));

    std::size_t remaining_cells = cell_indices.size()-current_cell_index;
    if(remaining_cells<workset_size)
      workset_size = remaining_cells;

    // this is the true number of cells in this workset
    wkst->num_cells = workset_size;
    wkst->details(0).cell_local_ids.resize(workset_size);
    wkst->details(1).cell_local_ids.resize(workset_size);

    for(std::size_t cell=0;cell<workset_size; cell++,current_cell_index++) {

      wkst->details(0).cell_local_ids[cell] = local_cell_ids_a[cell_indices[current_cell_index]];
      wkst->details(1).cell_local_ids[cell] = local_cell_ids_b[cell_indices[current_cell_index]];

      for (std::size_t vertex = 0; vertex < Teuchos::as<std::size_t>(vertex_coordinates_a.dimension(1)); ++ vertex) {
	for (std::size_t dim = 0; dim < Teuchos::as<std::size_t>(vertex_coordinates_a.dimension(2)); ++ dim) {
          wkst->details(0).cell_vertex_coordinates(cell,vertex,dim) = vertex_coordinates_a(cell_indices[current_cell_index],vertex,dim);
          wkst->details(1).cell_vertex_coordinates(cell,vertex,dim) = vertex_coordinates_b(cell_indices[current_cell_index],vertex,dim);
        }
      }
    }

    Kokkos::View<int*,PHX::Device> cell_local_ids_k_0 = Kokkos::View<int*,PHX::Device>("Workset:cell_local_ids",wkst->details(0).cell_local_ids.size());
    Kokkos::View<int*,PHX::Device> cell_local_ids_k_1 = Kokkos::View<int*,PHX::Device>("Workset:cell_local_ids",wkst->details(1).cell_local_ids.size());
    for(std::size_t i=0;i<wkst->details(0).cell_local_ids.size();i++) 
      cell_local_ids_k_0(i) = wkst->details(0).cell_local_ids[i];
    for(std::size_t i=0;i<wkst->details(1).cell_local_ids.size();i++) 
      cell_local_ids_k_1(i) = wkst->details(1).cell_local_ids[i];
    wkst->details(0).cell_local_ids_k = cell_local_ids_k_0;
    wkst->details(1).cell_local_ids_k = cell_local_ids_k_1;

    // fill the BasisValues and IntegrationValues arrays
    std::size_t max_workset_size = needs_a.cellData.numCells();
    populateValueArrays(max_workset_size,true,needs_a,wkst->details(0)); // populate "side" values
    populateValueArrays(max_workset_size,true,needs_b,wkst->details(1),Teuchos::rcpFromRef(wkst->details(0)));

    wkst++;
  }

  return wkst;
}

template<typename ArrayT>
std::vector<panzer::Workset>::iterator
panzer::buildEdgeWorksets(const std::vector<std::size_t> & cell_indices,
                          const panzer::PhysicsBlock & pb_a,
	 	          const std::vector<std::size_t>& local_cell_ids_a,
		          const std::vector<std::size_t>& local_side_ids_a,
		          const ArrayT& vertex_coordinates_a,
                          const panzer::PhysicsBlock & pb_b,
	      	          const std::vector<std::size_t>& local_cell_ids_b,
		          const std::vector<std::size_t>& local_side_ids_b,
		          const ArrayT& vertex_coordinates_b,
                          std::vector<Workset>::iterator beg)
{
  using Teuchos::RCP;
  using Teuchos::rcp;

  WorksetNeeds needs_a;
  needs_a.cellData = pb_a.cellData();

  const std::map<int,RCP<panzer::IntegrationRule> >& int_rules_a  = pb_a.getIntegrationRules();
  for(std::map<int,RCP<panzer::IntegrationRule> >::const_iterator ir_itr = int_rules_a.begin();
      ir_itr != int_rules_a.end(); ++ir_itr)
    needs_a.int_rules.push_back(ir_itr->second);
  
  const std::map<std::string,Teuchos::RCP<panzer::PureBasis> >& bases_a = pb_a.getBases();
  for(std::map<std::string,Teuchos::RCP<panzer::PureBasis> >::const_iterator b_itr = bases_a.begin();
      b_itr != bases_a.end(); ++b_itr)
    needs_a.bases.push_back(b_itr->second);

  WorksetNeeds needs_b;
  needs_b.cellData = pb_b.cellData();

  const std::map<int,RCP<panzer::IntegrationRule> >& int_rules_b  = pb_b.getIntegrationRules();
  for(std::map<int,RCP<panzer::IntegrationRule> >::const_iterator ir_itr = int_rules_b.begin();
      ir_itr != int_rules_b.end(); ++ir_itr)
    needs_b.int_rules.push_back(ir_itr->second);
  
  const std::map<std::string,Teuchos::RCP<panzer::PureBasis> >& bases_b = pb_b.getBases();
  for(std::map<std::string,Teuchos::RCP<panzer::PureBasis> >::const_iterator b_itr = bases_b.begin();
      b_itr != bases_b.end(); ++b_itr)
    needs_b.bases.push_back(b_itr->second);

  return buildEdgeWorksets(cell_indices,
                           needs_a,pb_a.elementBlockID(),local_cell_ids_a,local_side_ids_a,vertex_coordinates_a,
                           needs_b,pb_b.elementBlockID(),local_cell_ids_b,local_side_ids_b,vertex_coordinates_b,
                           beg);
}

template<typename ArrayT>
Teuchos::RCP<std::map<unsigned,panzer::Workset> >
panzer::buildBCWorkset(const panzer::PhysicsBlock& pb_a,
                       const std::vector<std::size_t>& local_cell_ids_a,
                       const std::vector<std::size_t>& local_side_ids_a,
                       const ArrayT& vertex_coordinates_a,
                       const panzer::PhysicsBlock& pb_b,
                       const std::vector<std::size_t>& local_cell_ids_b,
                       const std::vector<std::size_t>& local_side_ids_b,
                       const ArrayT& vertex_coordinates_b)
{
  // Get b's needs.
  WorksetNeeds needs_b;
  {
    needs_b.cellData = pb_b.cellData();
    const std::map<int,Teuchos::RCP<panzer::IntegrationRule> >& int_rules = pb_b.getIntegrationRules();
    for (std::map<int,Teuchos::RCP<panzer::IntegrationRule> >::const_iterator ir_itr = int_rules.begin();
         ir_itr != int_rules.end(); ++ir_itr)
      needs_b.int_rules.push_back(ir_itr->second);  
    const std::map<std::string,Teuchos::RCP<panzer::PureBasis> >& bases= pb_b.getBases();
    for(std::map<std::string,Teuchos::RCP<panzer::PureBasis> >::const_iterator b_itr = bases.begin();
        b_itr != bases.end(); ++b_itr)
    needs_b.bases.push_back(b_itr->second);
  }
  // Get a and b workset maps separately, but don't populate b's arrays.
  const Teuchos::RCP<std::map<unsigned,panzer::Workset> >
    mwa = buildBCWorkset(pb_a, local_cell_ids_a, local_side_ids_a, vertex_coordinates_a),
    mwb = buildBCWorkset(needs_b, pb_b.elementBlockID(), local_cell_ids_b, local_side_ids_b,
                         vertex_coordinates_b, false /* populate_value_arrays */);
  TEUCHOS_ASSERT(mwa->size() == mwb->size());
  for (std::map<unsigned,panzer::Workset>::iterator ait = mwa->begin(), bit = mwb->begin();
       ait != mwa->end(); ++ait, ++bit) {
    panzer::Workset& wa = ait->second;
    // Copy b's details(0) to a's details(1).
    wa.other = Teuchos::rcp(new panzer::WorksetDetails(bit->second.details(0)));
    // Populate details(1) arrays so that IP are in order corresponding to details(0).
    populateValueArrays(wa.num_cells, true, needs_b, wa.details(1), Teuchos::rcpFromRef(wa.details(0)));
  }
  // Now mwa has everything we need.
  return mwa;
}

#endif
