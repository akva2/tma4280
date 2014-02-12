      SUBROUTINE FST (VECTOR, N, Z, NN)
      INTEGER   N, NN, INV
      LOGICAL   INVERSE
      REAL*8    VECTOR(1:N-1)
      COMPLEX*16 Z(0:NN-1)
C
      INVERSE = .TRUE.
      CALL FAST_SINE (VECTOR, N, INVERSE, Z, NN)
C
      RETURN
      END
C
      SUBROUTINE FSTINV (VECTOR, N, Z, NN)
      INTEGER   N, NN, INV
      LOGICAL   INVERSE
      REAL*8    VECTOR(1:N-1)
      COMPLEX*16 Z(0:NN-1)
C
      INVERSE = .FALSE.
      CALL FAST_SINE (VECTOR, N, INVERSE, Z, NN)
C
      RETURN
      END
C
      SUBROUTINE FAST_SINE (VECTOR, N, INVERSE, Z, NN)
      INTEGER   N, K, KK
      LOGICAL   INVERSE
      REAL*8    VECTOR(1:N-1)
      COMPLEX*16 Z(0:nn-1)

C**********************************************************
C     IF WE EXTEND THE INPUT VECTOR TO THE 2*N VECTOR
C     XE = [0,X(1),...,X(N-1),0,-X(N-1),-X(N-2),...,-X(1)]
C     THEN
C     SINE TRANSFORM = [ i/2 FFT(XE) ](1:N-1)
C**********************************************************

      KK = 0
      Z(KK)  = DCMPLX(0D0,0D0)
      KK = KK+1
      DO K = 1,N-1
         Z(KK) = DCMPLX(VECTOR(K),0D0)
         KK    = KK+1
      END DO
      Z(KK) = DCMPLX(0D0,0D0)
      KK = KK+1
      DO K = N-1,1,-1
         Z(KK) = DCMPLX((-1)*VECTOR(K),0D0)
         KK    = KK+1
      END DO

C*************************************************************
C     APPLY FAST FOURIER TRANSFORM AND EXTRACT SINE TRANSFORM
C*************************************************************
         
      CALL FFT(Z,2*N)
      do i=0,(2*n)-1
         Z(i)  = Z(i) * (0d0,5d-1)
      enddo
      do i=1,n-1
         VECTOR(i) = Z(i)
      enddo
      
C****************************************************
C       INVERSE SINE TRANSFORM = (2/N)SINE TRANSFORM
C****************************************************
      
      IF (INVERSE) THEN
         do i=1,n-1
            VECTOR(i) = (2 * VECTOR(i)) / N
         enddo
      END IF
      
      RETURN
      END 

C================================================================
C     COMPLEX FAST FOURIER TRANSFORM
C================================================================

      SUBROUTINE FFT(X,M)
      COMPLEX*16 X(0:M)
      INTEGER M
      
C     BIT REVERSE VARIABLES
      
      INTEGER    INDEX, RINDEX, Q, S, LOGCOUNTER
      COMPLEX*16  TEMP
      
C     UPDATE VARIABLES
      
      INTEGER     LEVEL, COLS, HALFLEVEL, J, K
      COMPLEX*16   OMEGA, CTEMP
      REAL*8      PI
      
      PI = 3.141592653589793238462643d0
      
C*********************************************
C       BIT REVERSE PERMUTATION OF THE VECTOR
C*********************************************

      DO INDEX = 0,M-1
         
C     BIT REVERSE THE INDEX
         
         LOGCOUNTER = M
         RINDEX     = 0
         Q          = INDEX
         DO while (LOGCOUNTER .gt. 1)             

            s = Q/2
            RINDEX = (2*RINDEX)+ (Q - (2*S))
            Q = S
            LOGCOUNTER = LOGCOUNTER/2
         END DO
         
C     SWAP ENTRIES
         
         IF (RINDEX .gt. INDEX) THEN
            TEMP        = X(INDEX)
            X(INDEX)    = X(RINDEX)
            X(RINDEX)   = TEMP
         END IF
      END DO
      
C******************************
C     PERFORM BUTTERFLY UPDATE
C******************************
      HALFLEVEL  = 1
      LEVEL      = 2
      DO while (LEVEL .le. M) 
         
         COLS    = M/LEVEL
         
C****************************************************************
C     APPLY BUTTERFLY UPDATE TO KTH ROW OF X(LEVEL:LEVEL_COLUMNS)
C****************************************************************
         DO K = 0,COLS-1
            DO J = 0,HALFLEVEL-1
               OMEGA = DCMPLX(COS(2*PI*J/LEVEL),(-1)*SIN(2*PI*J/LEVEL))
               CTEMP = OMEGA *  X((K*LEVEL)+ J +HALFLEVEL)
               X((K*LEVEL)+ J +HALFLEVEL) = X((K*LEVEL)+ J) - CTEMP
               X((K*LEVEL)+ J )           = X((K*LEVEL)+ J) + CTEMP
            END DO
         END DO
         
         HALFLEVEL  = LEVEL
         LEVEL      = LEVEL * 2
      END DO
      
      return
      end


