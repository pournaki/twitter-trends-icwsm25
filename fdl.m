%%% ================== FORCE DIRECTED LAYOUT ================== %%%
%%%
%%% Takes each retweet network from `./output/rtn` and computes a 
%%% 2D fdl and saves it in `./output/fdl` using the same name.
%%% 
%%% Format:
%%% | user_id | x     | y     |
%%% |---------+-------+-------|
%%% | str     | float | float |

networkdir = './output/rtn';
actual_dir = pwd;
pattern = '*rtn_pp.csv';
files = dir(fullfile(networkdir, pattern));
filenames = {files.name};
numFiles = length(filenames);

fdlfolder = '../fdl'

% Check if folder exists, create it only if it doesn't
if ~exist(fdlfolder, 'dir')
    mkdir(fdlfolder);
    fprintf('Created folder: %s\n', fdlfolder);
else
    fprintf('Folder already exists: %s\n', fdlfolder);
end

% mkdir('../fdl');

chdir(networkdir); % Change into the desired directory

% Simple progress display
fprintf('Processing files: 0%%');

% Loop through the files
for i = 1:numFiles
    % Update progress display
    if i > 1
        fprintf(repmat('\b', 1, numel(num2str(floor(100 * (i-1) / numFiles))) + 1));
    end
    fprintf('%d%%', floor(100 * i / numFiles));
    
    filename = filenames{i};
    opts = detectImportOptions(filename);
    edges = readtable(filename, opts);
    G = digraph(edges.source, edges.target, edges.weight);
    coord = force_coordinates(G);
    ff = split(filename, '_rtn_pp.csv');
    coord_f = ['../fdl/', ff{1}, '_fdl.csv'];
    tt = G.Nodes;
    tt.Properties.VariableNames = {'user_id'};
    tt2 = table(coord(:,1), coord(:,2));
    tt2.Properties.VariableNames = {'x', 'y'};
    coord_t = [tt, tt2];
    writetable(coord_t, coord_f);
end

fprintf('\nProcessing complete!\n');
chdir(actual_dir);

function coord = force_coordinates(G)
    % Extracts the coordinates of a force-directed layout from a graph G
    p = plot(G, 'layout', 'force', 'WeightEffect', 'inverse');
    coord = [p.XData; p.YData]';
end

