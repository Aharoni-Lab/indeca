% MATLAB script for batch spike inference using MLSpike pipeline

function v_out = truncate_trailing_zeros(v)
    % Find the last non-zero element
    idx = find(v ~= 0, 1, 'last');
    
    % Return truncated vector
    if isempty(idx)
        v_out = [];  % all zeros
    else
        v_out = v(1:idx);
    end
end

% Define paths
input_dir = './tests/output/data/common';
output_dir = './tests/output/data/mlspike';

% Create output directory if it doesn't exist
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

% List all .nc files
files = dir(fullfile(input_dir, '*.nc'));

% Iterate through each file
for i = 1:length(files)
    file_path = fullfile(input_dir, files(i).name);
    output_path = fullfile(output_dir, files(i).name);

    % Load Y variable (fluorescence traces)
    Y = ncread(file_path, 'Y');
    dT = ncread(file_path, 'dT');

    % Preallocate output matrix
    [nFrames, nCells] = size(Y);
    N_matrix = nan(nFrames, nCells);

    % Iterate through each cell's trace
    for cell_idx = 1:nCells
        trace = Y(:, cell_idx);
        trace = truncate_trailing_zeros(trace);
        trace = trace / mean(trace);

        % Set parameters
        psig = struct('presetflag', 'correlated', 'freqs', min(3, 1 / dT / 6));
        pax = struct('dt', dT, 'maxamp', 15, 'amin', 0.6, 'amax', 1, 'autosigmasettings', psig);

        % Run spike calibration
        try
            [tau amp sigmaest eventdesc] = spk_autocalibration(trace, pax);
        catch ME
            tau = [];
        end

        if isempty(tau)
            par = struct('dt', dT, 'finetune', struct('autosigmasettings', psig));
        else
            par = struct('dt', dT, 'F0', [], 'a', amp, 'tau', tau, 'finetune', struct('sigma', sigmaest));
        end

        % Run MLSpike
        try
            [n, P, LL] = tps_mlspikes(trace, par);
        catch ME
            warning(ME.identifier, 'ERROR: %s', ME.message);
            continue
        end

        % Store result
        N_matrix(1:length(n), cell_idx) = n;
    end

    % Save result as .nc file
    if isfile(output_path)
        delete(output_path);
    end
    nccreate(output_path, 'S', ...
             'Dimensions', {'frame', nFrames, 'unit_id', nCells}, ...
             'Datatype', 'double');
    ncwrite(output_path, 'S', N_matrix);
end