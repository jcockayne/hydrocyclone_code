% Import experimental dataset and re-format

% stim = [n_elec x n_stim x   1    x   1   ]
% meas = [n_elec x    1   x n_meas x   1   ]
% data = [   1   x n_stim x n_meas x n_time]

% stim(i,j,1,1) encodes current flow through electrode i under stimulus j.
%               0 = no flow
%               1 = outward flow (current sink)
%              -1 = inward flow (current source)
% meas(i,1,k,1) indicates whether electrode i is being measured in measurement configuration k
%               0 = not measured 
%               1 = measured
% data(1,j,k,l) stores the measurement made in configuration k for stimulus j at time l

function [stim,meas,data] = import_data()

raw = load('ST1trial3.DAT');

n_elec = 8; % number of electrodes
n_stim = 7; % number of current stimulation patterns
n_meas = 8; % number of measurement configurations
n_time = 49; % number of time points

stim = zeros(n_elec,n_stim,   1  ,   1  );
meas = zeros(n_elec,   1,  n_meas,   1  );
data = zeros(  1   ,n_stim,n_meas,n_time);

% To start, drive a current between the reference electrode (Electrode 1) and
% Electrode 2, then in term measure the voltages between Electrode 1 and
% Electrode 2, Electrode 1 and Electrode 3, ... , Electrode 1 and Electrode
% 8. Next, move to a drive circuit between Electrode 1 and Electrode 3, and
% measure the voltages between Electrode 1 and Electrode 2, Electrode 1 and
% Electrode 3, ... , Electrode 1 and Electrode 8. Continue, moving the second
% drive electrode around until at Electrode 8, and measure the voltages between
% Electrode 1 and Electrode 2, Electrode 1 and Electrode 3, ... , Electrode 1
% and Electrode 8.

for i = 1:n_elec % index of electrodes
    for j = 1:n_stim % index of current pattern
        for k = 1:n_meas % index of measurement configuration
            for l = 1:n_time % index of time
                stim(i,j,1,1) = (i == 1) ... % index of drive electrode 
                                         ... % current comes *in* (opposite direction to outward normal)
                                 - (i == (1 + j)); % index of reference electrode (earthed)
                                           % current goes *out* (same direction as outward normal)
                
                meas(i,1,k,1) = +(i == k); % index of measurement electrode (wrt reference electrode)
                
                % wlog the potential at the reference electrode is 0
                if k > 1
                    data(1,j,k,l) = raw(l, 7*(j-1) + k - 1 ); % potential difference wrt reference electrode
                end
            end
        end
    end
end

end







