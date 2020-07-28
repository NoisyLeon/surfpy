#
INST_DIR = ../bin
BIN  = itomo_sp_cu_shn

fflags = -O2 -ffixed-line-length-none
#fflags = -fast -Mextend -Msave
#fflags =  -fast

# if you don't use shared libraries comment the next line
#shlib = -R/opt/SUNWspro/lib

FFLAGS = $(DBG)  $(fflags) $(shlib)

CFLAGS = $(DBG)

#FC = ifort
#FC = gfortran
FC = /usr/bin/gfortran

DBG = 

FOBJS = tomo_sp_cu.o help.o init.o menu.o set_defaults.o \
	reader.o m1.o rejector.o rejedel.o perc_map.o swt_sph.o \
	line.o tria.o trass.o model.o spl.o

$(BIN) : $(FOBJS)
	$(FC) $(FFLAGS) $(CFLAGS) $(FOBJS) -o $(BIN) $(LDLIBS)

install :: $(BIN)
	install  -s $(BIN) $(INST_DIR)

clean ::
	rm -f $(BIN) core $(FOBJS)
