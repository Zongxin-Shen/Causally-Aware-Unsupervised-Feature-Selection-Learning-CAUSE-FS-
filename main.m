clc; clear;
addpath('CAUSE-FS')
addpath('Datasets')

%% load data   
data = load('jaffe.mat');
X = data.fea;
Y = data.gnd;
c = length(unique(Y));
if size(X,2) ~= length(Y)
    X = X'; % d*n
end
initpar = InitPar(X,c);
X = mapminmax(double(X),0,1);

%% Parameter tuning
alpha = [0.001 0.01 0.1 1 10 100 1000];
beta = [10^5 10^5 10^6];
lambda = [0.001 0.01 0.1 1 10 100 1000];
total_combinations = length(alpha)*length(lambda)*length(beta);

best_ACC = 0;
best_NMI = 0;
best_params = [0, 0, 0];

feature_num = 20; % the number of selected features

fprintf('Starting parameter tuning with %d combinations...\n', total_combinations);

i=1;

for r_1 = 1:length(alpha)
    for r_2 = 1:length(beta)
        for r_3 = 1:length(lambda)
            fprintf('\nCurrent progress: %d/%d\n', i, total_combinations); 
            
            a = alpha(r_1);
            b = beta(r_2);
            l = lambda(r_3);
            % Run CAUSE-FS and select features
            [W] = CAUSE_FS(X,a,b,l,initpar,30);

            W_score = sum(W .* W,2);
            [~,index] = sort(W_score,'descend');
            indx = index(1:feature_num);
            X_fs = X(indx,:);   

            % Perform K-means clustering 50 times and report average performance
            AC = zeros(50, 1);
            MIhat = zeros(50, 1);
            for iter_c = 1:50 
                pre_labels = kmeans(X_fs',c,'Start','kmeans++');
                [ACC, NMI] = CalcMeasure(pre_labels, Y);
                AC(iter_c, 1) = ACC;
                MIhat(iter_c, 1) = NMI;
            end
            
            mean_ACC = mean(AC);
            mean_NMI = mean(MIhat);
            
            if mean_ACC > best_ACC || (mean_ACC == best_ACC && mean_NMI > best_NMI)
                best_ACC = mean_ACC;
                best_NMI = mean_NMI;
                best_params = [a, b, l];
            end

            i = i+1;
        end
    end
end

%% Display final results
fprintf('\nParameter tuning completed!\n');
fprintf('Best parameters found:\n');
fprintf('alpha = %.3f, beta = %.3f, lambda = %.3f\n', best_params(1), best_params(2), best_params(3));
fprintf('Best results:\n');
fprintf('ACC = %.4f, NMI = %.4f\n', best_ACC, best_NMI);

