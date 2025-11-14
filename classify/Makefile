#
# M A K E F I L E
#

dummy:
	echo No target specified

clean:
	rm -f *.log*

$(eval _INPUT = `cygpath -w $(INPUT)`)
$(eval _OUTPUT = `cygpath -w $(OUTPUT)`)

# YCBCRI is the default index
INDEX?=ycbcri

classify:
	python classify.py --file $(_INPUT) --output $(_OUTPUT) --index $(INDEX)