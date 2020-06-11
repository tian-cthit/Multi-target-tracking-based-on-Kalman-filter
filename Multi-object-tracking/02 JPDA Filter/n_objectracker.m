classdef n_objectracker
    %N_OBJECTRACKER is a class containing functions to track n object in
    %clutter. 
    %Model structures need to be called:
    %sensormodel: a structure specifies the sensor parameters
    %           P_D: object detection probability --- scalar
    %           lambda_c: average number of clutter measurements per time
    %           scan, Poisson distributed --- scalar 
    %           pdf_c: clutter (Poisson) intensity --- scalar
    %           intensity_c: clutter (Poisson) intensity --- scalar
    %motionmodel: a structure specifies the motion model parameters
    %           d: object state dimension --- scalar
    %           F: function handle return transition/Jacobian matrix
    %           f: function handle return predicted object state
    %           Q: motion noise covariance matrix
    %measmodel: a structure specifies the measurement model parameters
    %           d: measurement dimension --- scalar
    %           H: function handle return transition/Jacobian matrix
    %           h: function handle return the observation of the object
    %           state 
    %           R: measurement noise covariance matrix
    
    properties
        gating      %specify gating parameter
        reduction   %specify hypothesis reduction parameter
        density     %density class handle
    end
    
    methods
        
        function obj = initialize(obj,density_class_handle,P_G,m_d,w_min,merging_threshold,M)
            %INITIATOR initializes n_objectracker class
            %INPUT: density_class_handle: density class handle
            %       P_D: object detection probability
            %       P_G: gating size in decimal --- scalar
            %       m_d: measurement dimension --- scalar
            %       wmin: allowed minimum hypothesis weight --- scalar
            %       merging_threshold: merging threshold --- scalar
            %       M: allowed maximum number of hypotheses --- scalar
            %OUTPUT:  obj.density: density class handle
            %         obj.gating.P_G: gating size in decimal --- scalar
            %         obj.gating.size: gating size --- scalar
            %         obj.reduction.w_min: allowed minimum hypothesis
            %         weight in logarithmic scale --- scalar 
            %         obj.reduction.merging_threshold: merging threshold
            %         --- scalar 
            %         obj.reduction.M: allowed maximum number of hypotheses
            %         used in TOMHT --- scalar 
            obj.density = density_class_handle;
            obj.gating.P_G = P_G;
            obj.gating.size = chi2inv(obj.gating.P_G,m_d);
            obj.reduction.w_min = log(w_min);
            obj.reduction.merging_threshold = merging_threshold;
            obj.reduction.M = M;
        end
        
        function estimates = GNNfilter(obj, states, Z, sensormodel, motionmodel, measmodel)
            %GNNFILTER tracks n object using global nearest neighbor
            %association 
            %INPUT: obj: an instantiation of n_objectracker class
            %       states: structure array of size (1, number of objects)
            %       with two fields: 
            %                x: object initial state mean --- (object state
            %                dimension) x 1 vector 
            %                P: object initial state covariance --- (object
            %                state dimension) x (object state dimension)
            %                matrix  
            %       Z: cell array of size (total tracking time, 1), each
            %       cell stores measurements of size (measurement
            %       dimension) x (number of measurements at corresponding
            %       time step)  
            %OUTPUT:estimates: cell array of size (total tracking time, 1),
            %       each cell stores estimated object state of size (object
            %       state dimension) x (number of objects)
            
            t = length(Z); %total tracking time
            estimates = cell(t, 1);
            n = numel(states);

            for k = 1:t
                m = size(Z{k}, 2);   % number of measurements in the current time step.
                z_ingate = cell(n, 1);
                meas_in_gate_bool = zeros(n, m);
  
                % for each object, gating, get the gated measurement for
                % all objects in z_ingate_all
                for i = 1:n
                    [z_ingate{i}, meas_in_gate_bool(i, :)] = obj.density.ellipsoidalGating(states(i), Z{k}, measmodel, obj.gating.size);     % gating
                end
                meas_in_gate_bool_all = logical(sum(meas_in_gate_bool, 1)); % index of measurement in the gate for all objects
                z_ingate_all = Z{k}(:, meas_in_gate_bool_all); %    measurement in the gate for all objects
                meas_in_gate_bool = meas_in_gate_bool(:, meas_in_gate_bool_all);
                m = size(z_ingate_all, 2);
                % Cost matrix:
                L = inf(n, m + n);
                for i = 1:n
                    % predicted likelihood for each measurement in the gate
                    for j = find(meas_in_gate_bool(i, :))
                        predicted_likelihood_log = GaussianDensity.predictedLikelihood(states(i),z_ingate_all(:, j),measmodel);
                        L(i, j) = -log(sensormodel.P_D/sensormodel.intensity_c) - predicted_likelihood_log;    % detection weights
                    end
                    L(i, m+i) = -log(1 - sensormodel.P_D);    % misdetection weights
                end 
                [col4row,~] = assign2D(L);

                % update, prediction:
                for i = 1:n
                    if col4row(i) <= m   % if object i has a detection
                        states(i) = GaussianDensity.update(states(i), z_ingate_all(:, col4row(i)), measmodel);
                    end
                    estimates{k}(:, i) = states(i).x;
                    states(i) = GaussianDensity.predict(states(i), motionmodel); 
                end
                
            end
        end
            
            
        function estimates = JPDAfilter(obj, states, Z, sensormodel, motionmodel, measmodel)
        %JPDAFILTER tracks n object using joint probabilistic data
        %association
        %INPUT: obj: an instantiation of n_objectracker class
        %       states: structure array of size (1, number of objects)
        %       with two fields: 
        %                x: object initial state mean --- (object state
        %                dimension) x 1 vector 
        %                P: object initial state covariance --- (object
        %                state dimension) x (object state dimension)
        %                matrix  
        %       Z: cell array of size (total tracking time, 1), each
        %       cell stores measurements of size (measurement
        %       dimension) x (number of measurements at corresponding
        %       time step)  
        %OUTPUT:estimates: cell array of size (total tracking time, 1),
        %       each cell stores estimated object state of size (object
        %       state dimension) x (number of objects)
            t = length(Z); %total tracking time
            estimates = cell(t, 1);
            n = numel(states);

            for k = 1:t
                m = size(Z{k}, 2);   % number of measurements in the current time step.
                z_ingate = cell(n, 1);
                meas_in_gate_bool = zeros(n, m);

                % for each object, gating,
                for i = 1:n
                    [z_ingate{i}, meas_in_gate_bool(i, :)] = obj.density.ellipsoidalGating(states(i), Z{k}, measmodel, obj.gating.size);     % gating
                end
                meas_in_gate_bool_all = logical(sum(meas_in_gate_bool, 1)); % index of measurement in the gate for all objects
                z_ingate_all = Z{k}(:, meas_in_gate_bool_all); %    measurement in the gate for all objects
                meas_in_gate_bool = meas_in_gate_bool(:, meas_in_gate_bool_all);
                m = size(z_ingate_all, 2);
                
                % Cost matrix:
                L = inf(n, m + n);
                for i = 1:n
                % predicted likelihood for each measurement in the gate
                    for j = find(meas_in_gate_bool(i, :))
                        predicted_likelihood_log = GaussianDensity.predictedLikelihood(states(i),z_ingate_all(:, j),measmodel);
                        L(i, j) = -log(sensormodel.P_D/sensormodel.intensity_c) - predicted_likelihood_log;    % detection weights
                        
                    end
                    L(i, m+i) = -log(1 - sensormodel.P_D);    % misdetection weights
                end 

                [col4rowBest,~]=kBest2DAssign(L,obj.reduction.M);   %  col4rowBest: n x k matrix, each colum is a hypothesis, each object is matached to a detection/misdetection
                H = size(col4rowBest, 2);   % number of hypothesis
                
                %   weight of each global hypothesis, prune
                w = zeros(H, 1);
                for h = 1:H
                    w(h) = -trace(L(:, col4rowBest(:,h)));
                end
                [w, ~] = normalizeLogWeights(w);
                multiHypotheses = 1:H;
                [w, multiHypotheses] = hypothesisReduction.prune(w, multiHypotheses, obj.reduction.w_min);
                [w, ~] = normalizeLogWeights(w);
                col4rowBest = col4rowBest(:, multiHypotheses);
                H = size(col4rowBest, 2);
                
                w
                k
                L
                col4rowBest
                % local hypothesis
                beta = zeros(n, H);
                for i = 1:n
                    for h = 1:H
                        beta(i, h) =  -L(i, col4rowBest(i, h)) + w(h);  
                    end
                end
                beta
                %   merge hypotheses for the same object
                states_update = struct('x',{},'P',{});
                for i = 1:n
                    for h = 1:H
                        if col4rowBest(i, h) <= m   % if object i in hypothesis h has a detection
                            z = z_ingate_all(:, col4rowBest(i, h));
                            states_update(h, i) = GaussianDensity.update(states(i), z, measmodel);
                        else
                            states_update(h, i) = states(i);
                        end
                    end

                    [w_i, ~] = normalizeLogWeights(beta(i, :)');
                    states(i) = GaussianDensity.momentMatching(w_i, states_update(:, i));

                    estimates{k}(:, i) = states(i).x;
                    states(i) = GaussianDensity.predict(states(i), motionmodel); 
                end

            end
            
        end     
             
    end
end

