clear;

% 检查并启动并行池
%if isempty(gcp('nocreate'))
%    parpool('local');
%end

% 输入文件夹路径
currentpath = pwd;
folder_path = currentpath;
output_path = './results';
if ~exist(output_path, 'dir')
    mkdir(output_path);
end

% 文件名和相关数据信息
files = {'cmupie_withlabels.txt', 'coil20_withlabels.txt', 'umist_withlabels.txt', 'coil40_withlabels.txt', 'coil100_withlabels.txt','jaffe_withlabels.txt'};
channels = [1, 1, 1, 3, 3, 1];
img_widths = [32, 128, 92, 128, 128, 32];
img_heights = [32, 128, 112, 128, 128, 32];
% 最佳参数组合
best_params = [2, 1; 7, 4; 2, 9; 9, 9; 7, 4; 1, 7];

for f_idx = 1:length(files)
    data = load(fullfile(folder_path, files{f_idx}));
    datalabels = data(:, end);
    data(:, end) = [];
    
    K = numel(unique(datalabels));
    img_width = img_widths(f_idx);
    img_height = img_heights(f_idx);
    num_channels = channels(f_idx);
    
    % 使用预定义的最佳参数
    X = best_params(f_idx, 1);
    Y = best_params(f_idx, 2);
    
    [original_num_samples, original_num_features] = size(data);
    single_sample = reshape(data(1, 1:img_width*img_height), img_height, img_width);
    pooled_sample = blockproc(single_sample, [X, Y], @(block_struct) max(block_struct.data(:)));
    pooled_data = zeros(original_num_samples, numel(pooled_sample(:)) * num_channels);

    for ch = 1:num_channels
        start_col = (ch-1)*img_width*img_height + 1;
        end_col = ch*img_width*img_height;

        for sample_idx = 1:original_num_samples
            single_sample = reshape(data(sample_idx, start_col:end_col), img_height, img_width);
            pooled_sample = blockproc(single_sample, [X, Y], @(block_struct) max(block_struct.data(:)));
            pooled_data_start = (ch-1)*numel(pooled_sample(:)) + 1;
            pooled_data_end = ch*numel(pooled_sample(:));
            pooled_data(sample_idx, pooled_data_start:pooled_data_end) = pooled_sample(:)';
        end
    end
    
    DM = pdist2(pooled_data, pooled_data, 'cosine');
    idx = TorqueClustering1file(DM, K);
    NMI = nmi(idx, datalabels);
    ACC = accuracy(datalabels, idx) / 100;
    
    % 保存每个数据集的结果
    save_name = fullfile(output_path, strcat(files{f_idx}(1:end-4), '_results.txt'));
    fileID = fopen(save_name, 'w');
    fprintf(fileID, 'Best ACC achieved with X: %d and Y: %d, NMI: %.3f ACC: %.3f\n', X, Y, NMI, ACC);
    fclose(fileID);
end
