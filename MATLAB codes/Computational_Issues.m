%% Single hazard with increasing functionality states

% clr
% 
% T = 2000; dt = 1; Nsims = 25000;
% [RESULT] = Comp_Issues_3States(T, dt, Nsims, 'No');
% [RESULT] = Comp_Issues_4States(T, dt, Nsims, 'No');
% [RESULT] = Comp_Issues_5States(T, dt, Nsims, 'No');
% [RESULT] = Comp_Issues_6States(T, dt, Nsims, 'No');
% [RESULT] = Comp_Issues_7States(T, dt, Nsims, 'No');
% [RESULT] = Comp_Issues_8States(T, dt, Nsims, 'No');
% [RESULT] = Comp_Issues_9States(T, dt, Nsims, 'No');
% [RESULT] = Comp_Issues_10States(T, dt, Nsims, 'No');

%% Multihazards with increasing functionality states

clr

% mean = 365 days, 95% = 1093.44, 5% = 18.722

T = 2000; dt = 1; Nsims = 25000; T_int = 18.722;
% [res_fin] = Comp_Issues_3States_MH(T_int, T, dt, Nsims);
% [res_fin] = Comp_Issues_4States_MH(T_int, T, dt, Nsims);
% [res_fin] = Comp_Issues_5States_MH(T_int, T, dt, Nsims);
% [res_fin] = Comp_Issues_6States_MH(T_int, T, dt, Nsims);
% [res_fin] = Comp_Issues_7States_MH(T_int, T, dt, Nsims);
% [res_fin] = Comp_Issues_8States_MH(T_int, T, dt, Nsims);
% [res_fin] = Comp_Issues_9States_MH(T_int, T, dt, Nsims);
[res_fin] = Comp_Issues_10States_MH(T_int, T, dt, Nsims);


