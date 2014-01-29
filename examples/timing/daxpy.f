      program daxpytest
      include 'mpif.h'
      parameter (n = @SIZE@)
      real*8 a(n), b(n), t1, t2, dt, dt1, r
      integer loop
      call MPI_INIT (ierr)
      call MPI_COMM_RANK (MPI_COMM_WORLD, myid, ierr)
      call MPI_COMM_SIZE (MPI_COMM_WORLD, nproc, ierr)
      do l=1,k
         a(l) = dble(l+1)
      enddo
      do j=1,n
         b(l) = dble(l+1)
      enddo
      loop = 5
      t1 = MPI_WTIME()
      do iter=1,loop
          call daxpy_flagged(a,b,n)
      enddo
      t2 = MPI_WTIME()
      dt = t2 - t1
      dt1 = dt/(2*n)
      dt1 = dt1/loop
      r   = 1.E-6/dt1
      write (6,*) 'daxpy :(n)= ',n
      write (6,*) 'r=',r
      call MPI_FINALIZE(ierr)
      stop
      end

      subroutine daxpy_flagged(a,b,n)
c-------------------------------------------------------------
c matrix-matrix product
c      = a*b
c-------------------------------------------------------------
      real*8 a(n), b(n)
      real*8 alpha
      integer flag
      flag = @FLAG@
      alpha = 1
      if (flag .eq. 0) then
         call daxpy_std(alpha,a,b,n)
      else if (flag .eq. 1) then
          call daxpy(n,alpha,a,1,b,1)
      else if (flag .eq. 2) then
          call dot_std(a,a,n)
      else if (flag .eq. 3) then
          call dot_std(a,b,n)
      else
          write(6,*)'incorrect flag in daxpy'
          stop
      endif
      return
      end

      subroutine daxpy_std(a,x,y,n)
c---------------------------------------------------------
c daxpy
c = x = x+a*y
c "standard" case
c---------------------------------------------------------
      real*8 a, x(n), y(n)
      do i = 1, n
         y(i) = y(i) + a*x(i)
      enddo
      return
      end
      
      subroutine dot_std(x,y,n)
c---------------------------------------------------------
c inner product 
c = result = x^Ty 
c "standard" case
c---------------------------------------------------------
      real*8 a, x(n), y(n)
      a = 0;
      do i = 1, n
         a = a + x(i)*y(i);
      enddo
      x(1) = a;
      return
      end

