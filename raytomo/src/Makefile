#
INST_DIR = ../bin
BIN  = tomo_sp_cu_s

fflags = -O2 -ffixed-line-length-none
#fflags = -fast -Mextend -Msave
#fflags =  -fast -Mextend

# if you don't use shared libraries comment the next line
#shlib = -R/opt/SUNWspro/lib

FFLAGS = $(DBG)  $(fflags) $(shlib)

CFLAGS = $(DBG)

#FC = pgf77
FC = /usr/bin/gfortran
#FC = gfortran

DBG = 

FOBJS = tomo_sp_cu.o help.o init.o menu.o set_defaults.o reader.o \
	m1.o rejector.o rejedel.o perc_map.o swt_sph.o line.o \
	tria.o driver_2.o trass.o model.o spl.o cell_a.o \
	fresnel.o function.o matrix.o ec.o \
	res_anal.o res_form.o res_tria.o res_outp.o

$(BIN) : $(FOBJS)
	$(FC) $(FFLAGS) $(CFLAGS) $(FOBJS) -o $(BIN) $(LDLIBS)

install :: $(BIN)
	install  -s $(BIN) $(INST_DIR)

clean ::
	rm -f $(BIN) core $(FOBJS)
