#
BIN=lf_mhr_predict_earth
 
FFLAGS = -ffixed-line-length-none 
#FFLAGS =  -e
FC = gfortran
CFLAGS = 
 
DBG = 
 
$(BIN) : lf_mhr_predict_earth.o azidl.o spline.o  project.o new_path.o integr.o
	$(FC) $(FFLAGS) $(CFLAGS)  lf_mhr_predict_earth.o  azidl.o spline.o project.o new_path.o integr.o -o $(BIN) $(LDFLAGS) $(LDLIBS)
clean :
	rm -f $(BIN) *.o
