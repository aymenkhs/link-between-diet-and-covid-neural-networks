header = {'num_arch'; 'num_layers'; 'layer_1'; 'layer_2'; 'layer_3'; 'layer_4'; 'best_epoch';'best_train_performance'; 'best_test_performance'; 'train_fct'}; %dummy header

fid = fopen('results.csv', 'w');

num_outputs = length(header);

for i = 1:num_outputs

    fprintf(fid, '%s,', header{i});
end
fprintf(fid, '\n');


% import data
data = importdata('final_data.csv');
data = data.data;

intakes = data(:,1:36);
deaths = data(:, 37);
cases = data(:, 38);

clear data;

input = intakes';
target = deaths';

% parameters
num_architectures = 100;
max_layer = 4;
max_neurons = 20;


for n = 1 : num_architectures

    num_arch = int2str(n);

    % initialize first training to init best_tr

    % different possible train functions
    train_functions = {'trainrp', 'trainscg', 'traincgp', 'trainlm'};

    [net, hidden_layers] = generate_network(max_layer, max_neurons, train_functions{1});

    num_layers = length(hidden_layers);

    % format hidden layers to be able to write it
    hidden_layers = [hidden_layers,  zeros(1, max_layer - length(hidden_layers))];

    best_tr = ntrain(fid, n, num_layers, num_outputs, net, hidden_layers, input, target);


    % rest of the functions
    for i_t = 2: length(train_functions)

            net.trainFcn = train_functions{i_t};

            net = init(net);

            tr = ntrain(fid, n, num_layers, num_outputs, net, hidden_layers, input, target);

            % save best record
            if tr.best_tperf < best_tr.best_tperf

                best_tr = tr;
                num_conf = (n-1) * 4 + i_t;
            end
    end


    name = strcat('arch_', int2str(num_conf));
    name = strcat(name, '_');
    name = strcat(name, best_tr.trainFcn);
    n;
    saveas(plotperform(best_tr), fullfile('plots', name),'png');
end

fid(close)
clear
