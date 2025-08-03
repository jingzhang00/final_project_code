function [Data, Data_sh, att, x0_all, data, dt] = batch_load_LASA_dataset_DS(sub_sample, nb_trajectories, modelIdx)

names = {'Angle','BendedLine','CShape','DoubleBendedLine','GShape',...
         'heee','JShape','JShape_2','Khamesh','Leaf_1',...
         'Leaf_2','Line','LShape','NShape','PShape',...
         'RShape','Saeghe','Sharpc','Sine','Snake',...
         'Spoon','Sshape','Trapezoid','Worm','WShape','Zshape',...
         'Multi_Models_1', 'Multi_Models_2', 'Multi_Models_3','Multi_Models_4'};

modelName = names{modelIdx};
D = load(['DataSet/' modelName],'demos','dt');
dt = D.dt;
demos = D.demos;
N = length(demos);
att = [0 0]';
Data = []; x0_all = [];
trajectories = randsample(N, nb_trajectories)';
for l=1:nb_trajectories
    % Check where demos end and shift
    id_traj = trajectories(l);
    data{l} = [demos{id_traj}.pos(:,1:sub_sample:end); demos{id_traj}.vel(:,1:sub_sample:end)];
    Data = [Data data{l}];
    x0_all = [x0_all data{l}(1:2,20)];
end
Data_sh = Data;

end