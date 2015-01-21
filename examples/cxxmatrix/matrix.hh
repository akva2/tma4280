// An extremely simple matrix template
//
// Written by Arne Morten Kvarving. NO rights reserved.
//
// This code is in the public domain.

#pragma once

#include <iostream>
#include <vector>

//! \brief Enum for data layout in memory
enum DataOrder {
  RowMajor, //!< Row major - C ordering
  ColMajor  //!< Column major - Fortran ordering
};

//! \brief Matrix template
//! \param scalar Type of matrix elements
//! \param order Data layout in memory
  template<typename scalar, DataOrder order=RowMajor>
class Matrix {
  public:
    //! \brief Constructor
    //! \param rows Number of rows in matrix
    //! \param cols Number of columns in matrix
    Matrix(size_t rows, size_t cols) : m_rows(rows), m_cols(cols)
    {
      m_data.resize(m_rows*m_cols);
    }

    //! \brief Returns a reference to a matrix element
    //! \brief i Row number
    //! \brief j Column number
    //! \details This hides the data layout in memory from user.
    //!          Indices are zero based.
    inline scalar& operator()(size_t i, size_t j)
    {
      if (order == RowMajor)
        return m_data[i*m_cols+j];
      else
        return m_data[j*m_rows+i];
    }

    //! \brief Print matrix to a stream
    //! \param precision Precision for the scalars
    //! \param out The stream to print to. Defaults to std::cout
    void print(std::streamsize precision=0, std::ostream& out=std::cout)
    {
       std::streamsize oldprec;
       if (precision > 0) {
         oldprec = out.precision();
         out.precision(precision);
       }

       for (size_t i=0;i<m_rows;++i) {
         for (size_t j=0;j<m_cols;++j)
           out << (*this)(i,j) << " ";
         out << std::endl;
       }
       if (precision > 0)
         out.precision(oldprec);
    }
  private:
    size_t m_rows; 		//!< Number of rows in matrix
    size_t m_cols; 		//!< Number of columns in matrix
    std::vector<scalar> m_data; //!< The actual data
};
