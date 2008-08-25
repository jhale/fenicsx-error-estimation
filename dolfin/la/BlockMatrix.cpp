// Copyright (C) 2008 Kent-Andre Mardal.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2008-08-25

#include <stdexcept>
#include "BlockVector.h"
#include "BlockMatrix.h"
#include "DefaultFactory.h"
#include <iostream>

using namespace dolfin;

//-----------------------------------------------------------------------------
BlockMatrix::BlockMatrix(uint n_, uint m_, bool owner_): owner(owner_), n(n_), m(m_)
{
  matrices = new Matrix*[n*m]; 
  for (uint i=0; i<n; i++) 
  {
    for (uint j=0; j<m; j++) 
    {
      matrices[i*n + j] = new Matrix(); 
    }
  }
} 
//-----------------------------------------------------------------------------
BlockMatrix::~BlockMatrix() 
{
  if (owner) 
  {
    for (uint i=0; i<n; i++) 
    {
      for (uint j=0; j<m; j++) 
      {
        delete matrices[i*n + j];
      }
    }
  }
  delete [] matrices;  
}
//-----------------------------------------------------------------------------
const Matrix& BlockMatrix::mat(uint i, uint j) const 
{
  if (i >= n || j >= m) {  
    error("The index is out of range!");
  }
  return *(matrices[i*n+j]);
}
//-----------------------------------------------------------------------------
Matrix& BlockMatrix::mat(uint i, uint j) 
{
  if (i >= n || j >= m) {  
    error("The index is out of range!");
  }
  return *(matrices[i*n+j]);
}
//-----------------------------------------------------------------------------
dolfin::uint BlockMatrix::size(uint dim) const {
  if (dim==0) return n; 
  if (dim==1) return m; 
  error("BlockMatrix has rank 2!"); return 0; 
}
//-----------------------------------------------------------------------------
void BlockMatrix::zero()  
{
  for(uint i=0; i<n; i++) 
  {
    for(uint j=0; j<n; j++) 
    {
      this->mat(i,j).zero(); 
    }
  }
}
//-----------------------------------------------------------------------------
void BlockMatrix::apply()  
{
  for(uint i=0; i<n; i++) 
  {
    for(uint j=0; j<n; j++) 
    {
      this->mat(i,j).apply(); 
    }
  }
}
//-----------------------------------------------------------------------------
void BlockMatrix::disp(uint precision) const  
{
  for(uint i=0; i<n; i++) 
  {
    for(uint j=0; j<n; j++) 
    {
      std::cout <<"BlockMatrix("<<i<<","<<j<<"):"<<std::endl;  
      this->mat(i,j).disp(precision); 
    }
  }
}
//-----------------------------------------------------------------------------
void BlockMatrix::mult(const BlockVector& x, BlockVector& y, bool transposed) const  
{
  if (transposed) error("BlockMatrix::mult: transposed not implemented");  
  DefaultFactory factory; 
  GenericVector* vec; 
  vec = factory.createVector();
  for(uint i=0; i<n; i++) 
  {
    vec->init(y.vec(i).size()); 
    for(uint j=0; j<n; j++) 
    {
      this->mat(i,j).mult(x.vec(j), *vec);   
      y.vec(i) += *vec; 
    }
  }
}






