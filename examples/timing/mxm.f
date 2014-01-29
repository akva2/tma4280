      program mxmtest
      include 'mpif.h'
      parameter (m = @SIZE@); 
      parameter (k = @SIZE@); 
      parameter (n = @SIZE@); 
      real*8 a(m,k), b(k,n), c(m,n), t1, t2, dt, dt1, r
      integer*8 loop, m8, k8, n8

      m8 = m;
      k8 = k;
      n8 = n;

      call MPI_INIT (ierr)
      call MPI_COMM_RANK (MPI_COMM_WORLD, myid, ierr)
      call MPI_COMM_SIZE (MPI_COMM_WORLD, nproc, ierr)
      do l=1,k
         do i=1,m
            a(i,l) = dble(i+1)
         enddo
      enddo
      do j=1,n
         do l=1,k
            b(l,j) = dble(j+1)
         enddo
      enddo
      loop = 5
      t1 = MPI_WTIME(ierr)
      do iter=1,loop
          call mxm (a,m,b,k,c,n)
      enddo
      t2 = MPI_WTIME(ierr)
      dt = t2 - t1
      dt1 = dt/(m8*2*k8*n8)
      dt1 = dt1/loop
      r   = 1.E-6/dt1
      write (6,*) 'matrix-matrix :(m,k,n)= ',m,k,n
      write (6,*) 'r=',r
      call MPI_FINALIZE(ierr)
      stop
      end

      subroutine mxm(a,m,b,k,c,n)
c-------------------------------------------------------------
c matrix-matrix product
c      = a*b
c-------------------------------------------------------------
      real*8 a(m,k), b(k,n),c(m,n)
      real*8 alpha, beta
      integer flag
      flag = @FLAG@
      if (flag .eq. 0) then
         call mxm_std(a,m,b,k,c,n)
      else if (flag .eq. 1) then
        call mxm_unr(a,m,b,k,c,n)
      else if (flag .eq. 2) then
          alpha = 1.0
          beta = 0.0
          call dgemm('N','N',m,n,k,alpha,a,m,b,k,beta,c,m)
      else
          write(6,*)'incorrect flag in mxm'
          stop
      endif
      return
      end

      subroutine mxm_std(a,m,b,k,c,n)
c---------------------------------------------------------
c matrix-matrix product
c = a*b
c "standard" case
c---------------------------------------------------------
      real*8 a(m,k), b(k,n), c(m,n)
      do j=1,n
        do i=1,m
            c(i,j) = 0.0
            do l=1,k
                c(i,j) = c(i,j) + a(i,l)*b(l,j)
            enddo
         enddo
      enddo
      return
      end
      
      subroutine mxm_unr(a,m,b,k,c,n)
c---------------------------------------------------------
c matrix-matrix product
c = a*b
c unrolled case (k=10)
c---------------------------------------------------------
      real*8 a(m,k), b(k,n), c(m,n)
      do j=1,n
        do i=1,m
            c(i,j) = a(i,1)*b(1,j)
     +            + a(i,2)*b(2,j)
     +            + a(i,3)*b(3,j)
     +            + a(i,4)*b(4,j)
     +            + a(i,5)*b(5,j)
     +            + a(i,6)*b(6,j)
     +            + a(i,7)*b(7,j)
     +            + a(i,8)*b(8,j)
     +            + a(i,9)*b(9,j)
     +            + a(i,10)*b(10,j)
        enddo
      enddo
      return
      end
