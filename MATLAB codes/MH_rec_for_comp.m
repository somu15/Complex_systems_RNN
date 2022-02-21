clr

IMe = 0.01:0.05:2.0;
IMh = 10:5:125;
N_t = 250; Nsims = 1000;
count = 0;

for ii = 1:length(IMe)
    for jj = 1:length(IMh)
        req = zeros(2000,1);
        for kk = 1:N_t
            
            [res_fin, States] = Simulate_MH_rec(IMe(ii), IMh(jj), exprnd(150), 2000, 1, Nsims, 'E', 1);
            req = req+res_fin(2,:)';
            
        end
        
        MH_frag(:,ii,jj) = req/N_t;
        count = count + 1;
        progressbar(count/(length(IMe)*length(IMh)))
        
    end
end

%% Disfunctionality hazard

clr
dat = importdata('/Users/som/Dropbox/Complex_systems_RNN/Data/Multihazards/MH_EH_Frag_Exact.mat');


