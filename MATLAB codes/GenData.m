clr


IM = [50 65 80 95 110 125];%[0.01 0.1 0.16 0.3 0.61667 0.93333 1.25 1.65 1.8 1.95];

% IM = [0.01 0.05 0.1 0.16 0.3 0.45 0.61667 0.75 0.93333 1.1 1.25 1.65 1.8 1.85 1.95];

% IM = [0.01 0.05 0.1 0.16 0.3 0.37 0.45 0.57 0.61667 0.75 0.87 0.93333 1.1 1.25 1.45 1.57 1.65 1.8 1.85 1.95];

tr = 1:1000;

count = 1;
for ii = 1:length(IM)
    
    [Res, P11, P22] = Simulate_3state_rec(IM(ii), 'H', 1000, 1, 25000, 'Yes');
    min_ind = find(Res(1,:)==min(Res(1,:)));
    max_ind = find(Res(1,:)==max(Res(1,:)));
%     int = round((max_ind(1)-(min_ind(1)+3))/10);
%     indi = (min_ind(1)+3):int:max_ind(1);
    int = round((max_ind(1)-(min_ind(1)+123))/4);
    % indi = (min_ind(1)+3):int:max_ind(1);
    indi = [(4:10:600) (601:10:1000)];
    % indi = [(min_ind(1)+3) (min_ind(1)+33) (min_ind(1)+63) (min_ind(1)+93) (min_ind(1)+123) ((min_ind(1)+3):int:max_ind(1))];
    
    for jj = 1:length(indi)
        
        ind = indi(jj);
        data(count, 1) = Res(2,ind);
        data(count, 2) = Res(3,ind);
        data(count, 3) = Res(4,ind);
        data(count, 4) = IM(ii);
        data(count, 5) = P22(ind); % Res(1,ind);
        count = count + 1;
    end
    
    progressbar(ii/length(IM))
    
end

%% 

clr

% IM = 0.01:0.01:2; dIM = 0.01;
IM = 0.01:0.01:2; dIM = 0.01;
T = 1000; dt = 1; Nsims = 500;

for ii = 1:length(IM)
    
    [res] = Simulate_3state_rec(IM(ii),'E', T, dt, Nsims);
    
    % resilience(ii) = trapz(res(2,:),res(1,:));
    
    resil(ii,:) = res(1,:);
    
    progressbar(ii/length(IM))
    
end

load('/Users/som/Dropbox/MultiHazards/Disf_Hazard/Resilience Hazard Paper/Data/Hazards data/Charleston_Haz.mat');

hazr = exp(interp1(log(im),log(haz),log(IM)));
hazr_diff = abs(Differentiation(0.01,hazr));

hazr_diff = hazr_diff/trapz(IM,hazr_diff);

tr = 1:1:T;
for ii = 1:length(tr)
    
    resH(ii) = trapz(IM,(1-resil(:,ii)).*hazr_diff');
    
end

%% Hurricane

clr

IM = [10 25 40 60 75 85 95 105 115 125];
% IM = [10 25 40 60 75 80 85 90 95 100 105 110 115 120 125];
% IM = [10 25 40 45 50 55 60 65 70 75 80 85 90 95 100 105 110 115 120 125];

count = 1;
for ii = 1:length(IM)
    
    [Res, P11, P22] = Simulate_3state_rec(IM(ii), 'H', 1000, 1, 25000);
    min_ind = find(Res(1,:)==min(Res(1,:)));
    max_ind = find(Res(1,:)==max(Res(1,:)));
%     int = round((max_ind(1)-(min_ind(1)+3))/10);
%     indi = (min_ind(1)+3):int:max_ind(1);
    int = round((max_ind(1)-(min_ind(1)+123))/4);
    % indi = (min_ind(1)+3):int:max_ind(1);
    indi = [(4:10:600) (601:10:1000)];
    % indi = [(min_ind(1)+3) (min_ind(1)+33) (min_ind(1)+63) (min_ind(1)+93) (min_ind(1)+123) ((min_ind(1)+3):int:max_ind(1))];
    
    for jj = 1:length(indi)
        
        ind = indi(jj);
        data(count, 1) = Res(2,ind);
        data(count, 2) = Res(3,ind);
        data(count, 3) = Res(4,ind);
        data(count, 4) = IM(ii);
        data(count, 5) = Res(1,ind);
        count = count + 1;
    end
    
    progressbar(ii/length(IM))
    
end

% deleted 105 115 120 125

%% Hurricane P11 and P22 new

clr

IM = [10 25 40 60 75 85 95 105 115 125];
% IM = [10 25 40 60 75 80 85 90 95 100 105 110 115 120 125];
% IM = [10 25 40 45 50 55 60 65 70 75 80 85 90 95 100 105 110 115 120 125];

count = 1;
ind = 4:10:1000;

data = zeros(length(IM)*length(ind),5);
for ii = 1:length(IM)
    
    [Res, P11, P22] = Simulate_3state_rec(IM(ii), 'H', 1000, 1, 25000, 'Yes');
    
    data(((ii*length(ind)-length(ind)+1):ii*length(ind)),1) = [Res(2,ind)'];
    data(((ii*length(ind)-length(ind)+1):ii*length(ind)),2) = [Res(3,ind)'];
    data(((ii*length(ind)-length(ind)+1):ii*length(ind)),3) = [Res(4,ind)'];
    data(((ii*length(ind)-length(ind)+1):ii*length(ind)),4) = [IM(ii)*ones(length(Res(2,ind)),1)];
    data(((ii*length(ind)-length(ind)+1):ii*length(ind)),5) = [P22(ind)'];
        
    
    progressbar(ii/length(IM))
    
end

%% Earthquake data new

clr

% IMs = 0.1 

IM = 0.01:0.1:2;
IM = [IM 0.95 1.05 1.15 1.25 1.35 1.45 1.55 1.65 1.75 1.85 1.95];
tr = 1:1000;

count = 1;
dat = [];
for ii = 1:length(IM)
    
    Res = Simulate_3state_rec(IM(ii), 'E', 2000, 1, 2000);
    
    Res = Res';
    dat = [dat ; Res([4:25:1000],:)];
    
end

%% EQ reverse

clr

IM = 0.01:0.0663:1.99;
% IM = [IM 0.95 1.05 1.15 1.25 1.35 1.45];
% IM = [(0.01:0.1:0.5) (0.51:0.05:1) (1.01:0.07:1.5) (1.51:0.0612:2)];
% IM = [(0.01:0.075:0.5) (0.51:0.03:1) (1.01:0.07:1.5) (1.51:0.0612:2)];
tr = 1:1000;
tr_req = 4:10:1000;
% tr_ind = find(tr == tr_req);
count = 1;
for ii = 1:length(IM)
    
    Res = Simulate_3state_rec(IM(ii), 'E', 1000, 1, 25000);
    resil(ii,:) = 1-Res(1,:);
    P1 = Res(3,:);
    P2 = Res(4,:);
    
    for jj = 1:length(tr_req)
        
        ind = find(tr==tr_req(jj));
        dat(count,1) = Res(3,jj);
        dat(count,2) = Res(4,jj);
        dat(count,3) = Res(1,jj);
        count = count + 1;
        
    end
    
    progressbar(ii/length(IM))
    
end

%% Hurr recovery comparison

clr


IM = 125;%[50 65 80 95 110 125];%[0.01 0.1 0.16 0.3 0.61667 0.93333 1.25 1.65 1.8 1.95];

% IM = [0.01 0.05 0.1 0.16 0.3 0.45 0.61667 0.75 0.93333 1.1 1.25 1.65 1.8 1.85 1.95];

% IM = [0.01 0.05 0.1 0.16 0.3 0.37 0.45 0.57 0.61667 0.75 0.87 0.93333 1.1 1.25 1.45 1.57 1.65 1.8 1.85 1.95];

tr = 1:1000;

count = 1;
for ii = 1:length(IM)
    [Res, P11, P22] = Simulate_3state_rec(IM(ii), 'H', 1000, 1, 25000, 'Yes');
    data(:, 1) = Res(2,:)';
    data(:, 2) = Res(3,:)';
    data(:, 3) = Res(4,:)';
    data(:, 4) = IM(ii)*ones(1000,1);
    data(:, 5) = Res(1,:)';
end

