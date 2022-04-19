% The script just to verify the min-rate results as python's script returns
% inconsistent results
close all
clear
clc

% D2D Environment settings
SETTING = 'A';
if SETTING == 'A'
    N_LINKS = 10;
    FIELD_LENGTH = 100;
    SHORTEST_DIRECTLINK = 10;
    LONGEST_DIRECTLINK = 20;
elseif SETTING == 'B'
    N_LINKS = 15;
    FIELD_LENGTH = 200;
    SHORTEST_DIRECTLINK = 20;
    LONGEST_DIRECTLINK = 30;
end
SETTING_STRING = sprintf('N%d_L%d_%d-%dm', N_LINKS, FIELD_LENGTH, SHORTEST_DIRECTLINK, LONGEST_DIRECTLINK);

TX_POWER_dBm = 30;
NOISE_dBm_Hz = -169;
BANDWIDTH = 5e6;

TX_POWER = 10^((TX_POWER_dBm-30)/10);
NOISE_POWER = 10^((NOISE_dBm_Hz-30)/10) * BANDWIDTH;

CHANNELS_FILENAME = sprintf('Data/g_test_%s.mat', SETTING_STRING);
GP_FILENAME = sprintf('Data/GP_%s.mat', SETTING_STRING);
REGULAR_FILENAME = sprintf("Data/Regular_MinRateAlloc_%s.mat", SETTING_STRING);
TRANSFER_FILENAME = sprintf("Data/Transfer_MinRateAlloc_%s.mat", SETTING_STRING);
AE_TRANSFER_FILENAME = sprintf("Data/AE_Transfer_MinRateAlloc_%s.mat", SETTING_STRING);


% Load channels and pre-computed allocations
load(CHANNELS_FILENAME);
load(GP_FILENAME);
load(REGULAR_FILENAME);
load(TRANSFER_FILENAME);
load(AE_TRANSFER_FILENAME);
n_layouts = size(g, 1);
assert ((size(g,2)==N_LINKS) && (size(g,3)==N_LINKS));
assert ((size(power_controls_all, 1)==n_layouts) && ...
       (size(regular, 1)==n_layouts) && ...
       (size(transfer, 1)==n_layouts) && ...
       (size(ae_transfer, 1)==n_layouts));
assert ((size(power_controls_all, 2)==N_LINKS) && ...
       (size(regular, 2)==N_LINKS) && ...
       (size(transfer, 2)==N_LINKS) && ...
       (size(ae_transfer, 2)==N_LINKS));

fprintf('Evaluating %d samples on %s setting\n', n_layouts, SETTING_STRING);

% Compute min rate
pc_all = containers.Map;
pc_all('gp') = power_controls_all;
pc_all('regular') = regular;
pc_all('transfer') = transfer;
pc_all('ae_transfer') = ae_transfer;

min_rates_all = containers.Map;
for method_cell={'gp', 'regular', 'transfer', 'ae_transfer'}
    method = method_cell{1};
    pc = pc_all(method);
    min_rates = [];
    for layout = 1:n_layouts
        rates = compute_rates(pc(layout,:), squeeze(g(layout,:,:)), TX_POWER, NOISE_POWER, BANDWIDTH);
        min_rate = min(rates);
        min_rates = [min_rates, min_rate];
    end
    min_rates_all(method) = min_rates;
end

% Result analysis
for method_cell={'gp', 'regular', 'transfer', 'ae_transfer'}
    method = method_cell{1};
    fprintf('%s min rate: %.3f Mbps\n', method, mean(min_rates_all(method))/1e6);
end

% Function for computing rates
function r = compute_rates(power, channel, tx_power, noise_power, bandwidth) 
    n_links = size(channel, 1);
    assert ((size(power, 1)==1) && (size(power, 2)==n_links));
    assert ((size(channel, 1)==n_links) && (size(channel, 2)==n_links));
    signals = (power' .* diag(channel)) * tx_power;
    interferences = ((channel.*(1-eye(n_links)))*power') * tx_power + noise_power;
    r = bandwidth * log2(1+signals ./ interferences);
    assert ((size(r, 1)==n_links) && (size(r,2)==1));
    r = r';
end

