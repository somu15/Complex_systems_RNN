clear all
clc

% t1 = zeros(1000,1);
% t2 = zeros(1000,1);
% tr = zeros(1000,1);
% for ii = 1:10
%     t = [0;data((100*(ii-1)+1):(100*(ii-1)+100),1);1000];
%     p1 = [0;data((100*(ii-1)+1):(100*(ii-1)+100),2);1];
%     p2 = [0;data((100*(ii-1)+1):(100*(ii-1)+100),3);1];
%     rec = [0;data((100*(ii-1)+1):(100*(ii-1)+100),5);1];
%     r = rand(100,1);
%     [C,ia] = unique(p1);
%     t1((100*(ii-1)+1):(100*(ii-1)+100)) = interp1(C,t(ia),r);
%     [C,ia] = unique(p2);
%     t2((100*(ii-1)+1):(100*(ii-1)+100)) = interp1(C,t(ia),r);
%     [C,ia] = unique(rec);
%     tr((100*(ii-1)+1):(100*(ii-1)+100)) = interp1(C,t(ia),r);
% end
% data_t = [t1 t2 tr];

t1 = zeros(1000,1);
t2 = zeros(1000,1);
tr = zeros(1000,1);
p1 = [0;data(:,3);1];
p2 = [0;data(:,4);1];
rec = [0;data(:,1);1];
t = [0;data(:,2);1000];
r = rand(100,1);
[C,ia] = unique(p1);
t1 = interp1(C,t(ia),r);
[C,ia] = unique(p2);
t2 = interp1(C,t(ia),r);
[C,ia] = unique(rec);
tr = interp1(C,t(ia),r);

data_t = [t1 t2 tr];