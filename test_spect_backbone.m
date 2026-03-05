clear all, close all;
addpath('fw')
addpath(genpath('utils'))
%% Read File
data_dir = 'data/';
% paths = { 'plaza2', 'city10000', 'intel', 'grid3D','input_INTEL',  'single_drone', 'tiers'};
% paths = {'plaza1', 'mrclam7', 'all_nclt_sessions'};
% paths = {'ais2klinik', 'sphere2500'};
% paths = {'sphere2500'};
% paths = {'single_drone'};
% paths = { 'ais2klinik', ...
%          'all_nclt_sessions', ...
%          'city10000', ...
%          'grid3D', ...
%          'input_INTEL',  ...
%          'intel', ...
%          'mrclam7',...
%          'plaza1', ...
%          'plaza2',...
%          'single_drone',...
%          'sphere2500',...
%          'tiers'
         % };
paths = { 'MIT_snl', ...
         'intel_snl', ...
         'M3500_snl', ...
         'city10k_snl', ...
        'all_nclt_sessions', ...
        'ais2klinik', ...
         'city10000', ...
         'grid3D', ...
         'input_INTEL',  ...
         'intel', ...
         'mrclam7',...
         'plaza1', ...
         'plaza2',...
         'sphere2500',...
         'tiers',...
          'single_drone'
        %  'TUM-room.txt',...
        % 'TUM-desk.txt',...
        % 'TUM-computer-T.txt',...
        % 'TUM-computer-R.txt',...
        % 'Replica-REProom1_100.txt',...
        % 'Replica-REProom1.txt',...
        % 'Replica-REProom0_100.txt',...
        % 'Replica-REProom0.txt',...
        % 'Replica-REPoffice1_100.txt',...
        % 'Replica-REPoffice1.txt',...
        % 'Replica-REPoffice0_100.txt',...
        % 'Replica-REPoffice0.txt',...
        % 'MipNerf-room.txt',...
        % 'MipNerf-kitchen.txt',...
        % 'MipNerf-garden.txt',...
        % 'IMC-temple.txt',...
        % 'IMC-rome.txt',...
        % 'IMC-gate.txt',...
        % 'bal-93.txt',...
        % 'bal-392.txt',...
        % 'bal-1934.txt'
    };

set_weights_to_1 = false;

for path_idx =1:length(paths)
    path = paths{path_idx};   
    fprintf(path+"\n")
    [odom_edges, lc_edges, num_nodes] = read_edge_file(strcat(data_dir, path));
    if set_weights_to_1
        odom_edges(:,3) = 1;
        lc_edges(:,3) = 1;
    end
    edge_list=[odom_edges; lc_edges];
    % G = graph(edge_list(:,1), edge_list(:,2), edge_list(:,3), num_nodes);
    % most_connected_subgraph = mode(conncomp(G));
    % if max(conncomp(G)) ~= 1
    %     edge_list = edge_list(find(conncomp(G)==most_connected_subgraph),:);
    %     G = graph(edge_list(:,1), edge_list(:,2), edge_list(:,3), num_nodes);
    % end
    % [min_st_backbone, min_st_lc]= split_mst([odom_edges; lc_edges], num_nodes);
    % 
    % % negate the weights of the odom and lc edges and then use split mst to gat max spanning tree
    % max_odom_edges = odom_edges;
    % max_odom_edges(:,3) = -max_odom_edges(:,3);
    % max_lc_edges = lc_edges;
    % max_lc_edges(:,3) = -max_lc_edges(:,3);
    % 
    % [max_st_backbone, max_st_lc] = split_mst([max_odom_edges; max_lc_edges], num_nodes);
    % max_st_backbone(:,3) = -max_st_backbone(:,3);
    % max_st_lc(:,3) = -max_st_lc(:,3);
    % 


    parameter_sets_fixed = {};
    parameter_sets_fixed{end+1}= struct('name', 'Madow Fixed', 'rounding', 'madow'); % naieve with krylove (eigs) solver
    
    % parameter_sets_spectral_backbone = {};
    % parameter_sets_spectral_backbone{end+1}= struct('name', 'Madow Spectral BB', 'rounding', 'madow'); % naieve with 

    parameter_sets_spectral_backbone_effR = {};
    parameter_sets_spectral_backbone_effR{end+1}= struct('name', 'MadowSpectralBBeffRMaxST', 'rounding', 'madow'); % naieve with 


    parameter_sets_spectral_backbone_unweightedeffR = {};
    parameter_sets_spectral_backbone_unweightedeffR{end+1}= struct('name', 'MadowSpectralBBeffRUnweightedST', 'rounding', 'madow'); % naieve with 

    % [spectral_backbone, spectral_lc] = find_spectral_backbone([odom_edges; lc_edges], num_nodes);
    [spectral_backbone_effRunweighted, spectral_lc_effRunweighted] = find_spectral_backbone_effR([odom_edges; lc_edges], num_nodes, true, false);

    [spectral_backbone_effR, spectral_lc_effR] = find_spectral_backbone_effR([odom_edges; lc_edges], num_nodes, true, true);

    pct_min = (num_nodes-1)/ (size(lc_edges,1) + size(odom_edges,1));
   
    %% Solve using fixed edges
    
    pct_candidates_fixed = linspace(0.0,1.0,5);
    fprintf("Running MAC w/ odom backbone\n")
    results_fixed = solve_mac(true, pct_candidates_fixed, odom_edges, lc_edges, num_nodes, parameter_sets_fixed);

   fprintf("Running MAC w/ spectral backbone effR minst\n")
    results_specteffRunweighted = solve_mac(true, pct_candidates_fixed, spectral_backbone_effRunweighted, spectral_lc_effRunweighted, num_nodes, parameter_sets_spectral_backbone_unweightedeffR);
   fprintf("Running MAC w/ spectral backbone effR maxst\n")
    results_specteffR = solve_mac(true, pct_candidates_fixed, spectral_backbone_effR, spectral_lc_effR, num_nodes, parameter_sets_spectral_backbone_effR);

        max_f_round = 0;
    close all
    figure(1)
    grid on 
    hold on
    for i = 1:length(parameter_sets_fixed)
            name = parameter_sets_fixed{i}.name;
            name = matlab.lang.makeValidName(name);
            f_rounds = -[results_fixed.(name).f_round];
            max_f_round = max(max_f_round, max(f_rounds));
            figure(1)
            plot(pct_candidates_fixed*(1-pct_min)+pct_min, f_rounds, 'Marker','+', 'DisplayName', name)

    end
    for i = 1:length(parameter_sets_spectral_backbone_unweightedeffR)
            name = parameter_sets_spectral_backbone_unweightedeffR{i}.name;
            name = matlab.lang.makeValidName(name);
            f_rounds = -[results_specteffRunweighted.(name).f_round];
            max_f_round = max(max_f_round, max(f_rounds));
            figure(1)
            plot(pct_candidates_fixed*(1-pct_min)+pct_min, f_rounds,'Marker','+', 'DisplayName', name)

    end
    for i = 1:length(parameter_sets_spectral_backbone_effR)
            name = parameter_sets_spectral_backbone_effR{i}.name;
            name = matlab.lang.makeValidName(name);
            f_rounds = -[results_specteffR.(name).f_round];
            max_f_round = max(max_f_round, max(f_rounds));
            figure(1)
            plot(pct_candidates_fixed*(1-pct_min)+pct_min, f_rounds,'Marker','+', 'DisplayName', name)

    end
    legend;
    title(path)
    figure(1)
    solve_info_name = sprintf('mac_pgo_info/%s_info.mat', path);
    if set_weights_to_1
        solve_info_name = sprintf('mac_pgo_info_unweighted/%s_info.mat', path);
    end

    colors = ["r","b"];
    save(solve_info_name, 'results_specteffRunweighted', '-append')


end


