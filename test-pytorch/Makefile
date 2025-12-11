#
# M A K E F I L E
#

dummy:
	@echo No target specified

#
# This does not handle the case where there is a space in along the path
#
ifndef TEST
    TESTFLAG=
else
    TESTFLAG=--test `cygpath -w "$(TEST)"`
endif

ifeq ($(GRAPH),true)
	_GRAPHFLAG:=--graph
else
	_GRAPHFLAG:=
endif

ifndef EPOCHS
    _EPOCHS=
else
    _EPOCHS=--epochs $(EPOCHS)
endif


clean:
	@rm *.pt

train:
	$(eval _INPUT = `cygpath -w $(INPUT)`)
	$(eval _MODEL = `cygpath -w $(MODEL)`)
	python main.py -i $(_INPUT) -m $(_MODEL) $(_GRAPHFLAG) $(TESTFLAG) $(_EPOCHS)

test:
	$(eval _MODEL = `cygpath -w $(MODEL)`)
	python main.py -m $(_MODEL) $(TESTFLAG)
