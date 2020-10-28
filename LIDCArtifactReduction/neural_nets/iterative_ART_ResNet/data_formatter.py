from LIDCArtifactReduction.neural_nets.iterative_ART_ResNet.IterativeARTResNet import IterativeARTResNet


def output_data_formatter(actual_reconstructions, bad_sinograms, good_reconstructions):
    return {IterativeARTResNet.imgs_input_name: actual_reconstructions,
            IterativeARTResNet.sinos_input_name: bad_sinograms,
            IterativeARTResNet.resnet_name: good_reconstructions}

def input_data_decoder(data):
    return data[IterativeARTResNet.imgs_input_name], \
           data[IterativeARTResNet.sinos_input_name], \
           data[IterativeARTResNet.resnet_name]
