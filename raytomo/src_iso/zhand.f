      integer function hand ( sig, sip, uap )
      integer sig, address
      structure /fault/
        integer address
      end structure
      	
      structure /siginfo/
        integer si_signo
        integer si_code
        integer si_errno
        record /fault/ fault
      end structure

      record /siginfo/ sip

      address = sip.fault.address
      write (*,10) address
   10 format('Exception at hex address ', z8 )
      hand=0
      end
