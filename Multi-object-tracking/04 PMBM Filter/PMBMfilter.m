classdef PMBMfilter
    %PMBMFILTER is a class containing necessary functions to implement the
    %PMBM filter
    %Model structures need to be called:
    %    sensormodel: a structure specifies the sensor parameters
    %           P_D: object detection probability --- scalar
    %           lambda_c: average number of clutter measurements per time scan, 
    %                     Poisson distributed --- scalar
    %           pdf_c: value of clutter pdf --- scalar
    %           intensity_c: Poisson clutter intensity --- scalar
    %       motionmodel: a structure specifies the motion model parameters
    %           d: object state dimension --- scalar
    %           F: function handle return transition/Jacobian matrix
    %           f: function handle return predicted object state
    %           Q: motion noise covariance matrix
    %       measmodel: a structure specifies the measurement model parameters
    %           d: measurement dimension --- scalar
    %           H: function handle return transition/Jacobian matrix
    %           h: function handle return the observation of the target state
    %           R: measurement noise covariance matrix
    %       birthmodel: a structure array specifies the birth model (Gaussian
    %       mixture density) parameters --- (1 x number of birth components)
    %           w: weights of mixture components (in logarithm domain)
    %           x: mean of mixture components
    %           P: covariance of mixture components
    properties
        density %density class handle
        paras   %%parameters specify a PMBM
    end
    
    methods
        function obj = initialize(obj,density_class_handle,birthmodel)
            %INITIATOR initializes PMBMfilter class
            %INPUT: density_class_handle: density class handle
            %       birthmodel: a struct specifying the intensity (mixture)
            %       of a PPP birth model
            %OUTPUT:obj.density: density class handle
            %       obj.paras.PPP.w: weights of mixture components in PPP
            %       intensity --- vector of size (number of mixture
            %       components x 1) in logarithmic scale
            %       obj.paras.PPP.states: parameters of mixture components
            %       in PPP intensity struct array of size (number of
            %       mixture components x 1)
            %       obj.paras.MBM.w: weights of MBs --- vector of size
            %       (number of MBs (global hypotheses) x 1) in logarithmic 
            %       scale
            %       obj.paras.MBM.ht: hypothesis table --- matrix of size
            %       (number of global hypotheses x number of hypothesis
            %       trees). Entry (h,i) indicates that the (h,i)th local
            %       hypothesis in the ith hypothesis tree is included in
            %       the hth global hypothesis. If entry (h,i) is zero, then
            %       no local hypothesis from the ith hypothesis tree is
            %       included in the hth global hypothesis.
            %       obj.paras.MBM.tt: local hypotheses --- cell of size
            %       (number of hypothesis trees x 1). The ith cell contains
            %       local hypotheses in struct form of size (number of
            %       local hypotheses in the ith hypothesis tree x 1). Each
            %       struct has two fields: r: probability of existence;
            %       state: parameters specifying the object density
            
            obj.density = density_class_handle;
            obj.paras.PPP.w = [birthmodel.w]';
            obj.paras.PPP.states = rmfield(birthmodel,'w')';
            obj.paras.MBM.w = [];
            obj.paras.MBM.ht = [];
            obj.paras.MBM.tt = {};
        end
        
        function Bern = Bern_predict(obj,Bern,motionmodel,P_S)
            %BERN_PREDICT performs prediction step for a Bernoulli component
            %INPUT: Bern: a struct that specifies a Bernoulli component,
            %             with fields: r: probability of existence ---
            %                          scalar;
            %                          state: a struct contains parameters
            %                          describing the object pdf
            %       P_S: object survival probability
            
            Bern.r = Bern.r * P_S;
            Bern.state = obj.density.predict(Bern.state, motionmodel); 
        end
        
        function [Bern, lik_undetected] = Bern_undetected_update(obj,tt_entry,P_D)
            %BERN_UNDETECTED_UPDATE calculates the likelihood of missed
            %detection, and creates new local hypotheses due to missed
            %detection.
            %INPUT: tt_entry: a (2 x 1) array that specifies the index of
            %       local hypotheses. (i,j) indicates the jth local
            %       hypothesis in the ith hypothesis tree. 
            %       P_D: object detection probability --- scalar
            %OUTPUT:Bern: a struct that specifies a Bernoulli component,
            %       with fields: r: probability of existence --- scalar;
            %                    state: a struct contains parameters
            %                    describing the object pdf
            %       lik_undetected: missed detection likelihood --- scalar
            %       in logorithmic scale
            
            i = tt_entry(1);
            j = tt_entry(2);
            
            r = obj.paras.MBM.tt{i}(j).r;
            Bern.r = r*(1 - P_D)/(1 - r + r*(1 - P_D));
            Bern.state = obj.paras.MBM.tt{i}(j).state;
            lik_undetected = log(1 - r + r*(1 - P_D));
            
        end
        
        function lik_detected = Bern_detected_update_lik(obj,tt_entry,z,measmodel,P_D)
            %BERN_DETECTED_UPDATE_LIK calculates the predicted likelihood
            %for a given local hypothesis. 
            %INPUT: tt_entry: a (2 x 1) array that specifies the index of
            %       local hypotheses. (i,j) indicates the jth
            %       local hypothesis in the ith hypothesis tree.
            %       z: measurement array --- (measurement dimension x
            %       number of measurements)
            %       P_D: object detection probability --- scalar
            %OUTPUT:lik_detected: predicted likelihood --- (number of
            %measurements x 1) array in logarithmic scale 
            
            i = tt_entry(1);
            j = tt_entry(2);

            r = obj.paras.MBM.tt{i}(j).r;
            state = obj.paras.MBM.tt{i}(j).state;
            
            m = size(z, 2);
            lik_detected = zeros(m, 1);
            for i = 1:m
                predicted_likelihood_log = obj.density.predictedLikelihood(state,z(:, i),measmodel);
                lik_detected(i) = log(r*P_D) + predicted_likelihood_log;
            end
        end
        
        function Bern = Bern_detected_update_state(obj,tt_entry,z,measmodel)
            %BERN_DETECTED_UPDATE_STATE creates the new local hypothesis
            %due to measurement update. 
            %INPUT: tt_entry: a (2 x 1) array that specifies the index of
            %                 local hypotheses. (i,j) indicates the jth
            %                 local hypothesis in the ith hypothesis tree.
            %       z: measurement vector --- (measurement dimension x 1)
            %OUTPUT:Bern: a struct that specifies a Bernoulli component,
            %             with fields: r: probability of existence ---
            %                          scalar; 
            %                          state: a struct contains parameters
            %                          describing the object pdf 
            
            i = tt_entry(1);
            j = tt_entry(2);
            state = obj.paras.MBM.tt{i}(j).state;
            
            Bern.r = 1;
            Bern.state = obj.density.update(state, z, measmodel);
            
        end
        
        function obj = PPP_predict(obj,motionmodel,birthmodel,P_S)
            %PPP_PREDICT performs predicion step for PPP components
            %hypothesising undetected objects.
            %INPUT: P_S: object survival probability --- scalar     
            
            n = length(obj.paras.PPP.w);
            m = length([birthmodel.w]);
            
            obj.paras.PPP.w = [obj.paras.PPP.w + log(P_S); [birthmodel.w]'];
            
            % prediction of surviving undected objects
            for i = 1:n
                obj.paras.PPP.states(i) = obj.density.predict(obj.paras.PPP.states(i), motionmodel);
            end
            % new birth undetected objects
            for i = 1:m
                obj.paras.PPP.states(n+i, 1).x = birthmodel(i).x;
                obj.paras.PPP.states(n+i, 1).P = birthmodel(i).P;
            end

        end
        
        function [Bern, lik_new] = PPP_detected_update(obj,indices,z,measmodel,P_D,clutter_intensity)
            %PPP_DETECTED_UPDATE creates a new local hypothesis by
            %updating the PPP with measurement and calculates the
            %corresponding likelihood.
            %INPUT: z: measurement vector --- (measurement dimension x 1)
            %       P_D: object detection probability --- scalar
            %       clutter_intensity: Poisson clutter intensity --- scalar
            %       indices: boolean vector, if measurement z is inside the
            %       gate of mixture component i, then indices(i) = true
            %OUTPUT:Bern: a struct that specifies a Bernoulli component,
            %             with fields: r: probability of existence ---
            %             scalar;
            %             state: a struct contains parameters describing
            %             the object pdf
            %       lik_new: predicted likelihood of PPP --- scalar in
            %       logarithmic scale 
            
            n = sum(indices);   % number of PPP intensity components of which z is inside the gates.
            predicted_likelihood_log = zeros(n, 1);
            update = repmat(struct('x',{0}, 'P',{0}), [n, 1]); 
            states = obj.paras.PPP.states(indices);
            
            for i = 1:n
                state = states(i);
                update(i) = obj.density.update(state, z, measmodel);
                predicted_likelihood_log(i) = obj.density.predictedLikelihood(state,z,measmodel);
            end
            
            Rho = P_D * sum(exp(obj.paras.PPP.w(indices)) .* exp(predicted_likelihood_log));
            Bern.r = Rho/(Rho + clutter_intensity);
            lik_new = log(clutter_intensity + Rho);
            [predicted_likelihood_log_normalised, ~] = normalizeLogWeights(predicted_likelihood_log + obj.paras.PPP.w(indices));
            Bern.state = obj.density.momentMatching(predicted_likelihood_log_normalised, update);

        end
        
        function obj = PPP_undetected_update(obj,P_D)
            %PPP_UNDETECTED_UPDATE performs PPP update for missed detection.
            %INPUT: P_D: object detection probability --- scalar
            
            obj.paras.PPP.w = obj.paras.PPP.w + log(1 - P_D);
        end
        
        function obj = PPP_reduction(obj,prune_threshold,merging_threshold)
            %PPP_REDUCTION truncates mixture components in the PPP
            %intensity by pruning and merging
            %INPUT: prune_threshold: pruning threshold --- scalar in
            %       logarithmic scale
            %       merging_threshold: merging threshold --- scalar
            [obj.paras.PPP.w, obj.paras.PPP.states] = hypothesisReduction.prune(obj.paras.PPP.w, obj.paras.PPP.states, prune_threshold);
            if ~isempty(obj.paras.PPP.w)
                [obj.paras.PPP.w, obj.paras.PPP.states] = hypothesisReduction.merge(obj.paras.PPP.w, obj.paras.PPP.states, merging_threshold, obj.density);
            end
        end
        
        function obj = Bern_recycle(obj,prune_threshold,recycle_threshold)
            %BERN_RECYCLE recycles Bernoulli components with small
            %probability of existence, adds them to the PPP component, and
            %re-index the hypothesis table. If a hypothesis tree contains no
            %local hypothesis after pruning, this tree is removed. After
            %recycling, merge similar Gaussian components in the PPP
            %intensity
            %INPUT: prune_threshold: Bernoulli components with probability
            %       of existence smaller than this threshold are pruned ---
            %       scalar
            %       recycle_threshold: Bernoulli components with probability
            %       of existence smaller than this threshold needed to be
            %       recycled --- scalar
            
            n_tt = length(obj.paras.MBM.tt);
            for i = 1:n_tt
                idx = arrayfun(@(x) x.r<recycle_threshold & x.r>=prune_threshold, obj.paras.MBM.tt{i});
                if any(idx)
                    %Here, we should also consider the weights of different MBs
                    idx_t = find(idx);
                    n_h = length(idx_t);
                    w_h = zeros(n_h,1);
                    for j = 1:n_h
                        idx_h = obj.paras.MBM.ht(:,i) == idx_t(j);
                        [~,w_h(j)] = normalizeLogWeights(obj.paras.MBM.w(idx_h));
                    end
                    %Recycle
                    temp = obj.paras.MBM.tt{i}(idx);
                    obj.paras.PPP.w = [obj.paras.PPP.w;log([temp.r]')+w_h];
                    obj.paras.PPP.states = [obj.paras.PPP.states;[temp.state]'];
                end
                idx = arrayfun(@(x) x.r<recycle_threshold, obj.paras.MBM.tt{i});
                if any(idx)
                    %Remove Bernoullis
                    obj.paras.MBM.tt{i} = obj.paras.MBM.tt{i}(~idx);
                    %Update hypothesis table, if a Bernoulli component is
                    %pruned, set its corresponding entry to zero
                    idx = find(idx);
                    for j = 1:length(idx)
                        temp = obj.paras.MBM.ht(:,i);
                        temp(temp==idx(j)) = 0;
                        obj.paras.MBM.ht(:,i) = temp;
                    end
                end
            end
            
            %Remove tracks that contains no valid local hypotheses
            idx = sum(obj.paras.MBM.ht,1)~=0;
            obj.paras.MBM.ht = obj.paras.MBM.ht(:,idx);
            obj.paras.MBM.tt = obj.paras.MBM.tt(idx);
            if isempty(obj.paras.MBM.ht)
                %Ensure the algorithm still works when all Bernoullis are
                %recycled
                obj.paras.MBM.w = [];
            end
            
            %Re-index hypothesis table
            n_tt = length(obj.paras.MBM.tt);
            for i = 1:n_tt
                idx = obj.paras.MBM.ht(:,i) > 0;
                [~,~,obj.paras.MBM.ht(idx,i)] = unique(obj.paras.MBM.ht(idx,i),'rows','stable');
            end
            
            %Merge duplicate hypothesis table rows
            if ~isempty(obj.paras.MBM.ht)
                [ht,~,ic] = unique(obj.paras.MBM.ht,'rows','stable');
                if(size(ht,1)~=size(obj.paras.MBM.ht,1))
                    %There are duplicate entries
                    w = zeros(size(ht,1),1);
                    for i = 1:size(ht,1)
                        indices_dupli = (ic==i);
                        [~,w(i)] = normalizeLogWeights(obj.paras.MBM.w(indices_dupli));
                    end
                    obj.paras.MBM.ht = ht;
                    obj.paras.MBM.w = w;
                end
            end
            
        end
        
        function obj = PMBM_predict(obj,P_S,motionmodel,birthmodel)
            %PMBM_PREDICT performs PMBM prediction step.
            
            % MBM prediction
            H = numel(obj.paras.MBM.tt);
            for h = 1:H
                num_Bernoulli = numel(obj.paras.MBM.tt{h});
                for n = 1:num_Bernoulli
                    obj.paras.MBM.tt{h}(n) = Bern_predict(obj,obj.paras.MBM.tt{h}(n),motionmodel,P_S);
                end
            end
            
            % PPP prediction
            obj = PPP_predict(obj,motionmodel,birthmodel,P_S);

        end
        
        function obj = PMBM_update(obj,z,measmodel,sensormodel,gating,w_min,M)
            %PMBM_UPDATE performs PMBM update step.
            %INPUT: z: measurements --- array of size (measurement
            %       dimension x number of measurements)
            %       gating: a struct with two fields that specifies gating
            %       parameters: P_G: gating size in decimal --- scalar;
            %                   size: gating size --- scalar.
            %       wmin: hypothesis weight pruning threshold --- scalar in
            %       logarithmic scale
            %       M: maximum global hypotheses kept
            
            clutter_intensity = sensormodel.intensity_c;
            P_D = sensormodel.P_D;
            HT = numel(obj.paras.MBM.tt);    % number of hypotheses trees
            
            % gating for MBM
            z_ingate_MBM = cell(HT, 1);
            meas_in_gate_MBM = cell(HT, 1);
            meas_in_MBM_all_bool = cell(HT, 1);
            for ht = 1:HT
                num_Bernoulli = numel(obj.paras.MBM.tt{ht});
                z_ingate_MBM{ht} = cell(num_Bernoulli, 1);
                meas_in_gate_MBM{ht} = cell(num_Bernoulli, 1);
                
                for n = 1:num_Bernoulli
                    [z_ingate_MBM{ht}{n}, meas_in_gate_MBM{ht}{n}] = ...
                    obj.density.ellipsoidalGating(obj.paras.MBM.tt{ht}(n).state, z, measmodel, gating.size);
                end
                meas_in_MBM_all_bool{ht} = logical(sum([meas_in_gate_MBM{ht}{:}], 2));
            end
            meas_in_MBM_all_bool = logical(sum([meas_in_MBM_all_bool{:}], 2));
            
            % gating for PPP
            N = numel(obj.paras.PPP.states);
            z_ingate_PPP = cell(N, 1);
            meas_in_gate_PPP = false(size(z, 2), N);
            for i = 1:N
                [z_ingate_PPP{i}, meas_in_gate_PPP(:, i)] = obj.density.ellipsoidalGating(obj.paras.PPP.states(i), z, measmodel, gating.size);
            end
            
            meas_in_MBM_all_bool = logical(sum(meas_in_MBM_all_bool, 2));
            meas_in_MBM_all = z(:, meas_in_MBM_all_bool);
            
            meas_in_gate_all_bool = logical(sum([meas_in_MBM_all_bool, meas_in_gate_PPP], 2));
            meas_in_gate_all = z(:, meas_in_gate_all_bool);
            
            mk = size(meas_in_gate_all, 2);
%%            
            % MBM update
            Bern_undetected = cell(HT, 1);
            lik_undetected = cell(HT, 1);
            lik_detected = cell(HT, 1);
            Bernoulli_new = cell(HT, 1); % new Bernoullis at time k
            for ht = 1:HT
                num_Bernoulli = numel(obj.paras.MBM.tt{ht});
                Bern_undetected{ht} = cell(num_Bernoulli, 1);
                lik_undetected{ht} = zeros(num_Bernoulli, 1);
                lik_detected{ht} = cell(num_Bernoulli, 1);
                Bernoulli_new{ht} = cell(num_Bernoulli, 1);

                for n = 1:num_Bernoulli
                    
                    mk_B = size(z_ingate_MBM{ht}{n}, 2);    % number of measurement in the gate of this Bernoulli.                                 
                    [Bern_undetected{ht}{n}, lik_undetected{ht}(n)] = Bern_undetected_update(obj,[ht, n],P_D); % misdetection hypothesis:
                    Bernoulli_new{ht}{n}(1) = Bern_undetected{ht}{n};    % the first local hypothesis under each Bernoulli(local hypothesis) is the misdetection hypothesis
                    lik_detected{ht}{n} = Bern_detected_update_lik(obj,[ht, n],meas_in_MBM_all,measmodel,P_D);   % data association likelihood
                    % data association hypotheses:
                    for m = 1:mk_B
                        Bernoulli_new{ht}{n}(1 + m) = Bern_detected_update_state(obj,[ht, n],z_ingate_MBM{ht}{n}(:, m),measmodel);
                    end
                end
                
            end
            
            % PPP uppdate:
            % new birth:
            Bern_birth = cell(mk, 1);
            lik_birth = cell(mk, 1);
            for i = 1:mk
                dummy_Bern = struct('r', 0, 'state', 1);    %dummy bernoulli 
                indices = meas_in_gate_PPP(i, :);   % PPP components whose gates measurement i is in
                if any(indices(:) == 1) 
                    % for measurements in the gate of any PPP components,
                    % create birth Bernoullis.
                    [Bern_birth{i}, lik_birth{i}] = PPP_detected_update(obj,indices,meas_in_gate_all(:, i),measmodel,P_D,clutter_intensity);
                else
                    % for measurements not in the gate of any PPP
                    % components, create dummy Bernoullis.
                    [Bern_birth{i}, lik_birth{i}] = deal(dummy_Bern, clutter_intensity);
                end
            end
            
            % still undetected PPP:
            obj = PPP_undetected_update(obj,P_D);
            
            % new hypotheses tree:
            obj.paras.MBM.tt = cell(HT + mk, 1);   
            for ht = 1:HT
                obj.paras.MBM.tt{ht} = [Bernoulli_new{ht}{:}]';
            end
            
            for ht = 1:mk
                obj.paras.MBM.tt{HT+ht} = Bern_birth{ht};
            end

%%            
            if isempty(obj.paras.MBM.ht)
                obj.paras.MBM.ht = ones(1, mk);
                obj.paras.MBM.tt = Bern_birth;
                obj.paras.MBM.w = 0;    % log weight 
            else
                % cost matrix:
                H = size(obj.paras.MBM.ht, 1); % number of global hypotheses at time k-1
                HT = size(obj.paras.MBM.ht, 2); % number of hypotheses trees(components in a global hypotheses)

                L = cell(H, 1); % cost matrix
                col4rowBest = cell(H, 1);
                H_new = 0; % number of global hypotheses at time k
                MBM_w_new = [];  % global hypotheses weights array at time k
                
                for h = 1:H
                    L{h} = inf(mk, HT + mk);
                    % data association weights:
                    for i = 1:HT
                        if obj.paras.MBM.ht(h, i)~= 0   % if this tree exists in global hypothesis h.
                            Local_Bern_index = obj.paras.MBM.ht(h, i);
                            
                            for j = find(meas_in_gate_MBM{i}{Local_Bern_index}) % index of measurements in the gate of the Bernoulli
                                
                                lik_det = lik_detected{i}{Local_Bern_index}(j);
                                lik_undet = lik_undetected{i}(Local_Bern_index);
                                L{h}(j, i) = -(lik_det - lik_undet); 
                            end
                        end
                    end
                    
                    % new births weights
                    for m = 1:mk
                        L{h}(m, HT + m) = -lik_birth{m};
                    end
%                     L{h}

                    % K best assignments:
                    K = ceil(exp(obj.paras.MBM.w(h)) * M);
                    [col4rowBest{h},~] = kBest2DAssign(L{h},K); 
                    K_real = size(col4rowBest{h}, 2);

                    H_new = H_new + K_real;

                    % log weights for each new global hypothesis:
                    for k = 1:K_real                       
                        L_prior = obj.paras.MBM.w(h);
                        L_misdet = 0;
                        L_det = 0;
                        L_newB = 0;
                        Bern_index = [1:HT];
                        Bern_index = Bern_index(obj.paras.MBM.ht(h, :) ~= 0);
                        misdet_index = setdiff(Bern_index, col4rowBest{h}(:, k));
                        
                        for ht = misdet_index
                            
                            if obj.paras.MBM.ht(h, ht) ~= 0
                                L_misdet = L_misdet + lik_undetected{ht}(obj.paras.MBM.ht(h, ht));
                            end
                        end
                        
                        assign = col4rowBest{h}(:, k);
                        for i = 1:length(assign)
                            ht = assign(i);
                            if ht <= HT
%                                 col4rowBest{h}(:, k)
                                meas_index = col4rowBest{h}(:, k) == ht;
                                L_det = L_det + lik_detected{ht}{obj.paras.MBM.ht(h, ht)}(meas_index);
                            else
                                L_newB = L_newB + lik_birth{ht-HT};
                            end
                        end      
                        lik_global = L_prior + L_misdet + L_det + L_newB;

                        MBM_w_new = [MBM_w_new; lik_global];
                    end
                    

                end
                
                [MBM_w_new, ~] = normalizeLogWeights(MBM_w_new);
            

%%         
                % update look up table
                ht_new = zeros(H_new, HT + mk);
                H_ind = 0;
                for h = 1:H
                    K = size(col4rowBest{h}, 2); % number of data associations under global hypothesis h
                    for ht = 1:HT
                        ht_new(H_ind+1:H_ind+K, ht) = (obj.paras.MBM.ht(h, ht)-1)*(mk+1)+1;
                    end
                    % if a tree is not includeded in hypothesis h at time
                    % k-1, it's not included in all K hypotheses under h at
                    % time k:
                    trees_not_in_h = obj.paras.MBM.ht(h, :) == 0;
                    ht_new(H_ind+1:H_ind+K, trees_not_in_h) = 0;         
                    for i = 1:K   
                        H_ind = H_ind + 1;
                        for j = 1:mk
                            meas2ht = col4rowBest{h}(j, i);
                            if meas2ht <= HT
                                % data association:
                                ht_new(H_ind, meas2ht) = 1+j+(obj.paras.MBM.ht(h, meas2ht)-1)*(1+mk); % misdetection + j th measurement + (1+mk)*branches ahead
                            else
                                % new birth:
                                ht_new(H_ind, meas2ht) = 1; 
                            end
                        end
                    end
                end
                
    %%
                % pruning     
                ind = MBM_w_new > w_min;
                MBM_w_new = MBM_w_new(ind);
                ht_new = ht_new(ind, :);
                
                % capping
                if length(MBM_w_new) > M
                    [MBM_w_new, index] = maxk(MBM_w_new, M);
                    ht_new = ht_new(index, :);
                end
                [MBM_w_new, ~] = normalizeLogWeights(MBM_w_new);
    %%
                obj.paras.MBM.w = MBM_w_new;
    %%
                % prune hypotheses trees
                keep_ht_ind = (sum(ht_new, 1) ~= 0);
                obj.paras.MBM.tt = {obj.paras.MBM.tt{find(keep_ht_ind)}}';
                HT = numel(obj.paras.MBM.tt);
                ht_new = ht_new(:, find(keep_ht_ind));
                
                % prune local hypotheses do not appear in the look up table
                for ht = 1:HT
                    % remove unmatched local hypotheses

                    
                    tt_ind = unique(ht_new(:, ht));
             
                    tt_ind = tt_ind(tt_ind~=0);
                    
                    % re-index ht             
                    no_associ_Bern = setdiff([1:numel(obj.paras.MBM.tt{ht})], ht_new(:, ht));
                    no_associ_Bern = sort(no_associ_Bern, 'descend');
                    for i = no_associ_Bern
                        ind = ht_new(:, ht) > i;
                        ht_new(ind, ht) = ht_new(ind, ht) - 1;
                    end
                    obj.paras.MBM.tt{ht} = obj.paras.MBM.tt{ht}(tt_ind);
                end
                
                obj.paras.MBM.ht = ht_new;
                
            end
            
        end
        
        function estimates = PMBM_estimator(obj,threshold)
            %PMBM_ESTIMATOR performs object state estimation in the PMBM
            %filter
            %INPUT: threshold (if exist): object states are extracted from
            %       Bernoulli components with probability of existence no
            %       less than this threhold in Estimator 1. Given the
            %       probabilities of detection and survival, this threshold
            %       determines the number of consecutive misdetections
            %OUTPUT:estimates: estimated object states in matrix form of
            %       size (object state dimension) x (number of objects)
            %%%
            %First, select the multi-Bernoulli with the highest weight.
            %Second, report the mean of the Bernoulli components whose
            %existence probability is above a threshold. 
            
            [~, ind] = max(obj.paras.MBM.w);
            BestH = obj.paras.MBM.ht(ind, :);
            estimates = [];
            for i = 1:length(BestH)
                if BestH(i) ~= 0 && obj.paras.MBM.tt{i}(BestH(i)).r >= threshold
                    estimates = [estimates, obj.paras.MBM.tt{i}(BestH(i)).state.x];
                end
            end
            
        end
    
    end
end
