      program poisson
c==================================================================
c
c     solve the two-dimensional Poisson equation on a unit square 
C     using one-dimensional eigenvalue decompositions
c     and fast sine transforms
c
c     note: n needs to be a power of 2
c
c     einar m. rønquist
c     ntnu, october 2000
c
c===================================================================
      parameter (n  = 128)
      parameter (m  = n-1)
      parameter (nn = 4*n)
c
      real*8    diag(m), b(m,m), bt(m,m) 
      real*8    pi
      real*8    z(0:nn-1)
      real*4    tarray(2), t1, t2, dt

      h    = 1./n
      pi   = 4.*atan(1.)

      do i=1,m
         diag(i) = 2*(1-cos(i*pi/n))
      enddo
            
      do j=1,m
         do i=1,m
            b(i,j) = h*h
         enddo
      enddo
      
      do j=1,m
         call fst (b(1,j), n, z, nn)
      enddo 

      call transp (bt, b, m)
      do i=1,m
         call fstinv (bt(1,i), n, z, nn)
      enddo 

      do j=1,m
         do i=1,m
            bt(i,j) = bt(i,j)/(diag(i)+diag(j))
         enddo
      enddo

      do i=1,m
         call fst (bt(1,i), n, z, nn)
      enddo 
      call transp (b, bt, m)
      do j=1,m
         call fstinv (b(1,j), n, z, nn)
      enddo 

      umax = 0.0
      do j=1,m
         do i=1,m
            if (b(i,j) .gt. umax) umax = b(i,j)
         enddo
      enddo

      write(6,*) ' ' 
      write(6,*) umax

      stop
      end

      subroutine transp (at, a, m)
c====================================================
c     set at equal to the transpose of a 
c====================================================
      real*8 a(m,m), at(m,m)

      do j=1,m
         do i=1,m
            at(j,i) = a(i,j)
         enddo
      enddo
      return
      end



