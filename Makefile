NAMES=salinity retire.plan monitor.mercury term.structure milan.mort elec.temp sausage ustemp janka lidar fossil pig.weights age.income onions ragweed bpd trade.union ethanol scallop 
DATASETS=$(addprefix data/,$(addsuffix .txt,$(NAMES)))

all: $(DATASETS)
.PHONY: all clean

data/%.txt:
	http http://matt-wand.utsacademics.info/webspr/$(@F) > $@

clean: 
	rm -f $(DATASETS)
