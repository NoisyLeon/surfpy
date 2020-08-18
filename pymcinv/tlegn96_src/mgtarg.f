        subroutine mgtarg(i,name)
c---------------------------------------------------------------------c
c                                                                     c
c      COMPUTER PROGRAMS IN SEISMOLOGY                                c
c      VOLUME V                                                       c
c                                                                     c
c      SUBROUTINE: MGTARG                                             c
c                                                                     c
c      COPYRIGHT 1996 R. B. Herrmann                                  c
c                                                                     c
c      Department of Earth and Atmospheric Sciences                   c
c      Saint Louis University                                         c
c      221 North Grand Boulevard                                      c
c      St. Louis, Missouri 63103                                      c
c      U. S. A.                                                       c
c                                                                     c
c---------------------------------------------------------------------c
c-----
c       return the i'th command line argument
c
c       This version works on SUN, IBM RS6000
c-----
        implicit none
        integer i
        character name*(*)
            call getarg(i,name)
        return
        end
