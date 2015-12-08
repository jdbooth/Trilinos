// @HEADER
// ************************************************************************
//
//               Rapid Optimization Library (ROL) Package
//                 Copyright (2014) Sandia Corporation
//
// Under terms of Contract DE-AC04-94AL85000, there is a non-exclusive
// license for use of this work by or on behalf of the U.S. Government.
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
// Questions? Contact lead developers:
//              Drew Kouri   (dpkouri@sandia.gov) and
//              Denis Ridzal (dridzal@sandia.gov)
//
// ************************************************************************
// @HEADER

#ifndef ROL_LINESEARCH_H
#define ROL_LINESEARCH_H

/** \class ROL::LineSearch
    \brief Provides interface for and implements line searches.
*/

#include "Teuchos_RCP.hpp"
#include "Teuchos_ParameterList.hpp"
#include "ROL_Types.hpp"
#include "ROL_Vector.hpp"
#include "ROL_Objective.hpp"
#include "ROL_BoundConstraint.hpp"

namespace ROL { 

template<class Real>
class LineSearch {
private:

  ECurvatureCondition econd_;
  EDescent            edesc_;

  bool useralpha_;
  Real alpha0_;
  int maxit_;
  Real c1_;
  Real c2_;
  Real c3_;
  Real eps_;
  Real fmin_;         // smallest fval encountered
  Real alphaMin_;     // Alpha that yields the smallest fval encountered
  bool acceptMin_;    // Use smallest fval if sufficient decrease not satisfied
  bool itcond_;       // true if maximum function evaluations reached

  Teuchos::RCP<Vector<Real> > xtst_; 
  Teuchos::RCP<Vector<Real> > d_;
  Teuchos::RCP<Vector<Real> > g_;
  Teuchos::RCP<const Vector<Real> > grad_;

public:


  virtual ~LineSearch() {}

  // Constructor
  LineSearch( Teuchos::ParameterList &parlist ) : eps_(0.0) {
    // Enumerations
    edesc_ = StringToEDescent(parlist.sublist("Step").sublist("Line Search").sublist("Descent Method").get("Type","Quasi-Newton Method"));
    econd_ = StringToECurvatureCondition(parlist.sublist("Step").sublist("Line Search").sublist("Curvature Condition").get("Type","Strong Wolfe Conditions"));
    // Linesearc Parameters
    alpha0_    = parlist.sublist("Step").sublist("Line Search").get("Initial Step Size",1.0);
    useralpha_ = parlist.sublist("Step").sublist("Line Search").get("User Defined Initial Step Size",false);
    acceptMin_ = parlist.sublist("Step").sublist("Line Search").get("Accept Linesearch Minimizer",false);
    maxit_     = parlist.sublist("Step").sublist("Line Search").get("Function Evaluation Limit",20);
    c1_        = parlist.sublist("Step").sublist("Line Search").get("Sufficient Decrease Tolerance",1.e-4);
    c2_        = parlist.sublist("Step").sublist("Line Search").sublist("Curvature Condition").get("General Parameter",0.9);
    c3_        = parlist.sublist("Step").sublist("Line Search").sublist("Curvature Condition").get("Generalized Wolfe Parameter",0.6);

    fmin_      = std::numeric_limits<Real>::max();
    alphaMin_  = 0; 
    itcond_    = false;

    c1_ = ((c1_ < 0.0) ? 1.e-4 : c1_);
    c2_ = ((c2_ < 0.0) ? 0.9   : c2_);
    c3_ = ((c3_ < 0.0) ? 0.9   : c3_);
    if ( c2_ <= c1_ ) {
      c1_ = 1.e-4;
      c2_ = 0.9;
    }
    if ( edesc_ == DESCENT_NONLINEARCG ) {
      c2_ = 0.4;
      c3_ = std::min(1.0-c2_,c3_);
    }
  }

  virtual void initialize( const Vector<Real> &x, const Vector<Real> &s, const Vector<Real> &g,
                           Objective<Real> &obj, BoundConstraint<Real> &con ) {
    grad_ = Teuchos::rcp(&g, false);
    xtst_ = x.clone();
    d_    = s.clone();
    g_    = g.clone();
  }

  void setData(Real &eps) {
    eps_ = eps;
  }

  virtual bool status( const ELineSearch type, int &ls_neval, int &ls_ngrad, const Real alpha, 
                       const Real fold, const Real sgold, const Real fnew, 
                       const Vector<Real> &x, const Vector<Real> &s, 
                       Objective<Real> &obj, BoundConstraint<Real> &con ) { 
    Real tol = std::sqrt(ROL_EPSILON);

    // Check Armijo Condition
    bool armijo = false;
    if ( con.isActivated() ) {
      Real gs = 0.0;
      if ( edesc_ == DESCENT_STEEPEST ) {
        updateIterate(*d_,x,s,alpha,con);
        d_->scale(-1.0);
        d_->plus(x);
        gs = -s.dot(*d_);
      }
      else {
        d_->set(s);
        d_->scale(-1.0);
        con.pruneActive(*d_,grad_->dual(),x,eps_);
        gs = alpha*(grad_)->dot(d_->dual());
        d_->zero();
        updateIterate(*d_,x,s,alpha,con);
        d_->scale(-1.0);
        d_->plus(x);
        con.pruneInactive(*d_,grad_->dual(),x,eps_);
        gs += d_->dot(grad_->dual());
      }
      if ( fnew <= fold - c1_*gs ) {
        armijo = true;
      }
    }
    else {
      if ( fnew <= fold + c1_*alpha*sgold ) {
        armijo = true;
      }
    }

    // Check Maximum Iteration
    itcond_ = false;
    if ( ls_neval >= maxit_ ) { 
      itcond_ = true;
    }

    // Check Curvature Condition
    bool curvcond = false;
    if ( armijo && ((type != LINESEARCH_BACKTRACKING && type != LINESEARCH_CUBICINTERP) ||
                    (edesc_ == DESCENT_NONLINEARCG)) ) {
      if (econd_ == CURVATURECONDITION_GOLDSTEIN) {
        if (fnew >= fold + (1.0-c1_)*alpha*sgold) {
          curvcond = true;
        }
      }
      else if (econd_ == CURVATURECONDITION_NULL) {
        curvcond = true;
      }
      else {
        updateIterate(*xtst_,x,s,alpha,con);
        obj.update(*xtst_);
        obj.gradient(*g_,*xtst_,tol);
        Real sgnew = 0.0;
        if ( con.isActivated() ) {
          d_->set(s);
          d_->scale(-alpha);
          con.pruneActive(*d_,s,x);
          sgnew = -d_->dot(g_->dual());
        }
        else {
          sgnew = s.dot(g_->dual());
        }
        ls_ngrad++;
   
        if (    ((econd_ == CURVATURECONDITION_WOLFE)       
                     && (sgnew >= c2_*sgold))
             || ((econd_ == CURVATURECONDITION_STRONGWOLFE) 
                     && (std::abs(sgnew) <= c2_*std::abs(sgold)))
             || ((econd_ == CURVATURECONDITION_GENERALIZEDWOLFE) 
                     && (c2_*sgold <= sgnew && sgnew <= -c3_*sgold))
             || ((econd_ == CURVATURECONDITION_APPROXIMATEWOLFE) 
                     && (c2_*sgold <= sgnew && sgnew <= (2.0*c1_ - 1.0)*sgold)) ) {
          curvcond = true;
        }
      }
    }

    if(fnew<fmin_) {
      fmin_ = fnew;
      alphaMin_ = alpha;
    }

    if (type == LINESEARCH_BACKTRACKING || type == LINESEARCH_CUBICINTERP) {
      if (edesc_ == DESCENT_NONLINEARCG) {
        return ((armijo && curvcond) || itcond_);
      }
      else {
        return (armijo || itcond_);
      }
    }
    else {
      return ((armijo && curvcond) || itcond_);
    }
  }

  virtual void run( Real &alpha, Real &fval, int &ls_neval, int &ls_ngrad,
                    const Real &gs, const Vector<Real> &s, const Vector<Real> &x, 
                    Objective<Real> &obj, BoundConstraint<Real> &con ) = 0;

  virtual Real getInitialAlpha(int &ls_neval, int &ls_ngrad, const Real fval, const Real gs, 
                               const Vector<Real> &x, const Vector<Real> &s, 
                               Objective<Real> &obj, BoundConstraint<Real> &con) {
    Real val = 1.0;
    if (useralpha_) {
      val = alpha0_;
    }
    else {
      if (edesc_ == DESCENT_STEEPEST || edesc_ == DESCENT_NONLINEARCG) {
        Real tol = std::sqrt(ROL_EPSILON);
        // Evaluate objective at x + s
        updateIterate(*d_,x,s,1.0,con);
        obj.update(*d_);
        Real fnew = obj.value(*d_,tol);
        ls_neval++;
        // Minimize quadratic interpolate to compute new alpha
        Real denom = (fnew - fval - gs);
        Real alpha = ((denom > ROL_EPSILON) ? -0.5*gs/denom : 1.0);
        val = ((alpha > 1.e-1) ? alpha : 1.0);

        alpha0_ = val;
        useralpha_ = true;
      }
      else {
        val = 1.0;
      }
    }
    return val;
  }

  void updateIterate(Vector<Real> &xnew, const Vector<Real> &x, const Vector<Real> &s, Real alpha,
                     BoundConstraint<Real> &con ) {

    xnew.set(x);
    xnew.axpy(alpha,s);

    if ( con.isActivated() ) {
      con.project(xnew);
    }
  }

  bool useLocalMinimizer() {
    return itcond_ && acceptMin_;
  }
 
  bool takeNoStep() {
    return itcond_ && !acceptMin_;
  }

  // use this function to modify alpha and fval if the maximum number of iterations
  // are reached
  void setMaxitUpdate(Real &alpha, Real &fnew, const Real &fold) {
    // Use local minimizer
    if( itcond_ && acceptMin_ ) {
      alpha = alphaMin_;
      fnew = fmin_;
    }
    // Take no step
    else if(itcond_ && !acceptMin_) {
      alpha = 0;
      fnew = fold;
    }
  }
 

};

}

#include "ROL_LineSearchFactory.hpp"

#endif
