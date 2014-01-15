program hello
  include 'mpif.h'
  integer rank, size, ierror, tag
  integer status(MPI_STATUS_SIZE)
  character(12) message
  call MPI_INIT (ierror)
  call MPI_COMM_SIZE (MPI_COMM_WORLD, size, ierror)
  call MPI_COMM_RANK (MPI_COMM_WORLD, rank, ierror)
  tag = 100
  if (rank .eq. 0) then
    message = 'Hello world!'
    do i=1, size-1
      call MPI_SEND (message, 12, MPI_CHARACTER, i, tag, MPI_COMM_WORLD, ierror)
  enddo
  else
    call MPI_RECV (message, 12, MPI_CHARACTER, 0, tag, MPI_COMM_WORLD, status, ierror)
  endif
  print *, 'process', rank, ': ', message
  call MPI_Finalize (ierror)
end
