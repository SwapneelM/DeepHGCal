
source activate deepHGCal
export DEEPHGCAL=`pwd`
export DEEPJETCORE=`pwd`/../DeepJetCore

cd $DEEPJETCORE
source gpu_env.sh
cd $DEEPHGCAL

export PATH=$DEEPHGCAL/scripts:$PATH
export PYTHONPATH=$DEEPHGCAL/python:$DEEPHGCAL/DNN/modules:$PYTHONPATH
export PATH=$DEEPHGCAL/Converter/exe:$DEEPHGCAL/Converter/scripts:$PATH

