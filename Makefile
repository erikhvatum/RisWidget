# A gruesomely hacked Generic GNUMakefile

# Just a snippet to stop executing under other make(1) commands
# that won't understand these lines
ifneq (,)
This makefile requires GNU Make.
endif

LIBRARY = ris_widget.so
CPP_FILES := sipris_widgetRisWidget.cpp sipris_widgetcmodule.cpp $(wildcard *.cpp)
OBJS := $(patsubst %.cpp, %.o, $(CPP_FILES))
CC = g++
CPPFLAGS = -c -pipe -O2 -fPIC -std=c++0x -march=native -Wall -W -D_REENTRANT -DQT_NO_DEBUG -DQT_WIDGETS_LIB -DQT_GUI_LIB -DQT_CORE_LIB -I/usr/lib64/qt5/mkspecs/linux-g++ -I. -I/usr/include/qt5 -I/usr/include/qt5/QtWidgets -I/usr/include/qt5/QtGui -I/usr/include/qt5/QtCore -I/usr/include/qt5/QtOpenGL -I/usr/include/python3.3
LDFLAGS = -Wl,-O1 -std=c++0x -shared -lQt5Widgets -lQt5Gui -lQt5Core -lQt5OpenGL -lGL -lpthread cpp/*.o
RUNSIP = sip -c . -b ris_widget.sbf -I /usr/share/sip/PyQt5 -x VendorID -t WS_X11 -t Qt_5_2_0 -g ris_widget.sip

all: $(LIBRARY)

$(LIBRARY): .depend $(OBJS)
	$(CC) $(LDFLAGS) $(OBJS) -Wl,-soname,$(LIBRARY) -o $(LIBRARY)

depend: sipris_widgetRisWidget.cpp sipris_widgetcmodule.cpp .depend

.depend: cmd = $(CC) $(CPPFLAGS) -MM -MF depend $(var); cat depend >> .depend;
.depend:
	@echo "Generating dependencies..."
	@$(foreach var, $(CPP_FILES), $(cmd))
	@rm -f depend

-include .depend

# These are the pattern matching rules. In addition to the automatic
# variables used here, the variable $* that matches whatever % stands for
# can be useful in special cases.
%.o: %.cpp sipris_widgetRisWidget.cpp sipris_widgetcmodule.cpp sipAPIris_widget.h ris_widget.sbf
	$(CC) $(CPPFLAGS) -c $< -o $@

sipris_widgetRisWidget.cpp:
	$(RUNSIP)

sipris_widgetcmodule.cpp:
	$(RUNSIP)

sipAPIris_widget.h:
	$(RUNSIP)

ris_widget.sbf:
	$(RUNSIP)

clean:
	rm -f .depend $(LIBRARY) $(PROGRAM) $(OBJS) sipris_widgetRisWidget.cpp sipris_widgetcmodule.cpp sipAPIris_widget.h ris_widget.sbf

.PHONY: clean depend
