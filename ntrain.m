
function [tr] = ntrain(fid, num_arch, num_layers, num_outputs, net, hidden_layers, input, target)

    [net, tr] = train(net, input, target);

    
    % format

    result = [num_arch, num_layers, hidden_layers, tr.best_epoch, tr.best_perf, tr.best_tperf];
    

    for i = 1 : num_outputs

        if  i == num_outputs
            fprintf(fid,'%s,', net.trainFcn);
        else
            fprintf(fid,'%d,', result(i));
        end
    end
    fprintf(fid, '\n');
end