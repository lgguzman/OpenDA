! $Id: generate_covar.F90 1165 2011-09-14 13:38:14Z lnerger $
!BOP
!
! !Program: example_eofcovar_uv --- Compute covariance matrix from state trajectory
!
! !INTERFACE:
PROGRAM example_eofcovar_uv

! !DESCRIPTION:
! This programm computes a covariance matrix from a
! sequence of model states. The model states are stored as
! vectors in simple ASCII files. The matrix is decomposed in
! EOFs and stored in form of eigenvalues and eigenvectors.
! This example is for a univariate case, e.g. a single model
! field. 
!
! The matrix is generated by a singular value decomposition
! of the perturbation matrix of the model states about their
! mean state.
!
! !REVISION HISTORY:
! 2013-11 - L. Nerger - Initial coding

! !USES:
  IMPLICIT NONE
!EOP
  
! Local variables
  INTEGER :: i, iter              ! Counters
  INTEGER :: nstate               ! Size of model field
  INTEGER :: nfiles               ! Number of input files (=number of model states)
  CHARACTER(len=120) :: inpath, infile ! Path to and name stub of input files
  CHARACTER(len=120) :: outfile_eof, outfile_mstate, outfile_svals ! Names of output files
  REAL, ALLOCATABLE :: states(:, :)    ! Array holding model states
  CHARACTER(len=2) :: ensstr           ! String for ensemble member

  ! Output variables of EOF routine
  INTEGER :: status = 0           ! Output of EOF routine: status flag
  REAL :: stddev                  ! Output of EOF routine: multivariate STDDEV (not used here)
  REAL, ALLOCATABLE :: svals(:)   ! Output of EOF routine: Singular values
  REAL, ALLOCATABLE :: svecs(:,:) ! Output of EOF routine: Singular vectors
  REAL, ALLOCATABLE :: meanstate(:) ! Output of EOF routine: mean state


! ************************************************
! *** Configuration                            ***
! ************************************************

  ! Number of state files to be read
  nfiles = 5

  ! State dimension
  nstate = 4

  ! Path to and name of file holding model trajectory
  inpath = './inputs/'
  infile = 'fieldA_'

  ! Names of output files
  outfile_eof = 'eof_'             ! Files holding EOFs
  outfile_svals = 'svals.txt'      ! Files holding singular values
  outfile_mstate = 'meanstate.txt' ! Files holding mean state


! ************************************************
! *** Init                                     ***
! ************************************************

  WRITE (*,'(10x,a)') '*******************************************'
  WRITE (*,'(10x,a)') '*           example_EOFCovar_UV           *'
  WRITE (*,'(10x,a)') '*                                         *'
  WRITE (*,'(10x,a)') '*    Compute covariance matrix and mean   *'
  WRITE (*,'(10x,a)') '*     state from a sequence of states.    *'
  WRITE (*,'(10x,a)') '*                                         *'
  WRITE (*,'(10x,a)') '*      Example for univariate EOFs.       *'
  WRITE (*,'(10x,a)') '*                                         *'
  WRITE (*,'(10x,a)') '*   Write covar matrix as scaled eigen-   *'
  WRITE (*,'(10x,a)') '*    vectors and singular values into     *'
  WRITE (*,'(10x,a)') '*                  files                  *'
  WRITE (*,'(10x,a/)') '*******************************************'


! ************************
! *** Read state files ***
! ************************

  WRITE (*,'(/1x,a)') '------- Read states -------------'
  WRITE (*,*) 'Read states from files:  ',TRIM(inpath)//TRIM(infile),'*.txt'

  ALLOCATE(states(nstate, nfiles))

  read_in: DO iter = 1, nfiles

     WRITE (ensstr, '(i1)') iter
     OPEN(11, file = TRIM(inpath)//TRIM(infile)//TRIM(ensstr)//'.txt', status='old')
 
     DO i = 1, nstate
        READ (11, *) states(i, iter)
     END DO

     CLOSE(11)

  END DO read_in


! *************************************************
! *** Call routine to perform EOF decomposition ***
! *************************************************

  ALLOCATE(svals(nfiles))
  ALLOCATE(svecs(nstate, nfiles))
  ALLOCATE(meanstate(nstate))

  CALL sangoma_eofcovar(nstate, nfiles, 1, 1, 1, &
       1, 0, states, stddev, svals, svecs, meanstate, status)

  WRITE (*,'(5x,a)') 'Scaled singular values: '
  DO i = 1, nfiles-1
    WRITE (*, '(10x, i4, es12.3)') i, svals(i)
  END DO


! *********************************************************
! *** Write mean state and decomposed covariance matrix ***
! *********************************************************

  WRITE (*,'(/1x,a)') '------- Write decomposed covariance matrix and mean state -------------'

  ! *** Write singular values ***
  WRITE (*,*) 'Write singular vectors to file: ',TRIM(outfile_svals)
  OPEN(11, file = TRIM(outfile_svals), status='replace')
  DO i = 1, nfiles-1
     WRITE (11, *) svals(i)
  END DO
  CLOSE(11)

  ! *** Write EOFs ***
  WRITE (*,*) 'Write eofs to files: ',TRIM(outfile_eof),'*.txt'
  writing: DO iter = 1, nfiles-1

     WRITE (ensstr, '(i1)') iter
     OPEN(11, file = TRIM(outfile_eof)//TRIM(ensstr)//'.txt', status='replace')
 
     DO i = 1, nstate
        WRITE (11, *) svecs(i, iter)
     END DO

     CLOSE(11)

  END DO writing

  ! *** Write mean state ***
  WRITE (*,*) 'Write meanstate to file: ',TRIM(outfile_mstate)
  OPEN(11, file = TRIM(outfile_mstate), status='replace')
  DO i = 1, nstate
     WRITE (11, *) meanstate(i)
  END DO
  CLOSE(11)


! ********************
! *** Finishing up ***
! ********************

   DEALLOCATE(states, meanstate)
   DEALLOCATE(svals, svecs)

  WRITE (*,'(/1x,a/)') '------- END -------------'

END PROGRAM example_eofcovar_uv

