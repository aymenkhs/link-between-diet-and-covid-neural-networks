% function that generates random network architecture
function [network, hidden_layers] = generate_network(max_layer, max_neurons, train_function)
    
    % define the number of layers
    num_layers = randi([1, max_layer]);
  
    %define number of neurons on each hidden layer
    hidden_layers = [];
    for i = 1 : num_layers
        
        num_neurons = randi([1, max_neurons]);
        hidden_layers = [hidden_layers, num_neurons];
    end 
    
    net = feedforwardnet(hidden_layers, train_function);

    net.divideFcn = 'divideint';
    net.divideParam.trainRatio = 0.7;
    net.divideParam.valRatio = 0.15;
    net.divideParam.testRatio = 0.15;
    
    network = net;
end