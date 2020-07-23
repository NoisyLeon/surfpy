      SUBROUTINE HELP
      IMPLICIT NONE
c       help list   ==========

      write(*,*) '         GENERAL:'
      write(*,*) ' q[uit] or ex[it]..........quit program'
      write(*,*) ' h[elp] or ?...............print help menu to screen'
      write(*,*) ' me[nu]....................enter settings menu'
      write(*,*) ' v[iew]....................view settings'
      write(*,*) ' def[aults]................reset settings to defaults'
      write(*,*) ' 	INSIDE MENU:   '
      write(*,*) ' r or q or x...............return to the main program'
      write(*,*) ' v[iew]....................view settings'                  
      write(*,*) ' 1,2,...,11................see settings'                  
      return
      end
