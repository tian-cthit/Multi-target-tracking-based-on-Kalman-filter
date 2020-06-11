classdef singleobjectracker
    %SINGLEOBJECTRACKER is a class containing functions to track a single
    %object in clutter. 
    %Model structures need to be called:
    %sensormodel: a structure specifies the sensor parameters
    %           P_D: object detection probability --- scalar
    %           lambda_c: average number of clutter measurements per time
    %           scan, Poisson distributed --- scalar 
    %           pdf_c: clutter (Poisson) density --- scalar
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
            %INITIATOR initializes singleobjectracker class
            %INPUT: density_class_handle: density class handle
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
            %         --- scalar 
            
            obj.density = density_class_handle;
            obj.gating.P_G = P_G;
            obj.gating.size = chi2inv(obj.gating.P_G,m_d);
            obj.reduction.w_min = log(w_min);
            obj.reduction.merging_threshold = merging_threshold;
            obj.reduction.M = M;
        end
        
        function estimates = nearestNeighbourFilter(obj, state, Z, sensormodel, motionmodel, measmodel)
            %NEARESTNEIGHBOURFILTER tracks a single object using nearest
            %neighbor association 
            %INPUT: state: a structure with two fields:
            %                x: object initial state mean --- (object state
            %                dimension) x 1 vector 
            %                P: object initial state covariance --- (object
            %                state dimension) x (object state dimension)
            %                matrix  
            %       Z: cell array of size (total tracking time, 1), each
            %       cell stores measurements of  
            %            size (measurement dimension) x (number of
            %            measurements at corresponding time step) 
            %OUTPUT:estimates: cell array of size (total tracking time, 1),
            %       each cell stores estimated object state of size (object
            %       state dimension) x 1 
            
            t = length(Z); %total tracking time
            estimates = cell(t, 1);
            w_0 = 1 - sensormodel.P_D; 
            for i = 1:t
                [z_ingate, ~] = GaussianDensity.ellipsoidalGating(state, Z{i}, measmodel, obj.gating.size);
                
                if ~isempty(z_ingate)   %if there are measurements
                    predicted_likelihood_log = GaussianDensity.predictedLikelihood(state,z_ingate,measmodel);
                    [max_predicted_likelihood_log, I] = max(predicted_likelihood_log);
                    max_predicted_likelihood = exp(max_predicted_likelihood_log);
                    w_theta_max = sensormodel.P_D * max_predicted_likelihood / sensormodel.intensity_c;
                    
                    if w_theta_max > w_0
                        state = GaussianDensity.update(state, z_ingate(:, I), measmodel);   %update
                    end
                end
                
                estimates{i} = state.x;
                state = GaussianDensity.predict(state, motionmodel);    %prediction
            end
            
            
        end
        
        
        function estimates = probDataAssocFilter(obj, state, Z, sensormodel, motionmodel, measmodel)
            %PROBDATAASSOCFILTER tracks a single object using probalistic
            %data association 
            %INPUT: state: a structure with two fields:
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
            %       state dimension) x 1  
            
            t = length(Z); %total tracking time
            w_0_log = log(1 - sensormodel.P_D); 
            estimates = cell(t, 1);
            
            for i = 1:1:t
                [z_ingate, ~] = GaussianDensity.ellipsoidalGating(state, Z{i}, measmodel, obj.gating.size);
                if ~isempty(z_ingate)   %if there are measurement
                    % multiHypotheses
                    multiHypotheses(1) = state; % no detection hypothesis
                    for j = 1:size(z_ingate, 2)
                        multiHypotheses(j+1) = GaussianDensity.update(state, z_ingate(:, j), measmodel);  % detection hypotheses
                    end
                    multiHypotheses = multiHypotheses';
                    
                    detection_hypotheses_weights_log = GaussianDensity.predictedLikelihood(state,z_ingate,measmodel) + log(sensormodel.P_D / sensormodel.intensity_c);
                    hypotheses_weights_log = [w_0_log; detection_hypotheses_weights_log];
                    [log_w, ~] = normalizeLogWeights(hypotheses_weights_log);
                    [hypotheses_weights_log, multiHypotheses] = hypothesisReduction.prune(log_w, multiHypotheses, obj.reduction.w_min);
                    [log_w, ~] = normalizeLogWeights(hypotheses_weights_log);
                    state = GaussianDensity.momentMatching(log_w, multiHypotheses);
                end
                
                estimates{i} = state.x;
                state = GaussianDensity.predict(state, motionmodel);    %prediction
            end 
        end
        
        function estimates = GaussianSumFilter(obj, state, Z, sensormodel, motionmodel, measmodel)
            %GAUSSIANSUMFILTER tracks a single object using Gaussian sum
            %filtering
            %INPUT: state: a structure with two fields:
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
            %       state dimension) x 1  
            
            t = length(Z); %total tracking time
            w_0_log = log(1 - sensormodel.P_D); 
            estimates = cell(t, 1);
            
            state_pred = state; %%initial state
            hypothesesWeight = log(1); %initial weight
            for i = 1:t
                
                %for each hypothesis
                multiHypotheses_new = struct('x',{},'P',{});
                w_new = [];
                H = length(hypothesesWeight);
                for h = 1:H
                    [z_ingate, ~] = GaussianDensity.ellipsoidalGating(state_pred(h), Z{i}, measmodel, obj.gating.size);
                    detection_hypotheses_weights_log = GaussianDensity.predictedLikelihood(state_pred(h),z_ingate,measmodel) + ...
                            log(sensormodel.P_D / sensormodel.intensity_c);

                    % for each measurement create detection hypothesis
                    for j = 1:size(z_ingate, 2)
                        multiHypotheses_new(end + 1) = GaussianDensity.update(state_pred(h), z_ingate(:, j), measmodel);
                    end
                    w_new =[w_new; hypothesesWeight(h) + detection_hypotheses_weights_log]; % detection hypotheses weights
                    % no detection hypothesis & weights
                    multiHypotheses_new(end + 1) = state_pred(h);
                    w_new =[w_new; hypothesesWeight(h) + w_0_log];
                end
                multiHypotheses_new = multiHypotheses_new';

                [w_new, ~] = normalizeLogWeights(w_new);
                [hypotheses_weights_log, multiHypotheses_new] = hypothesisReduction.prune(w_new, multiHypotheses_new, obj.reduction.w_min);
                [w_new, ~] = normalizeLogWeights(hypotheses_weights_log);
                [w_new,multiHypotheses_new] = ...
                    hypothesisReduction.merge(w_new,multiHypotheses_new,obj.reduction.merging_threshold,obj.density);
                [w_new, ~] = normalizeLogWeights(w_new);
                [w_new, multiHypotheses_new] = hypothesisReduction.cap(w_new, multiHypotheses_new, obj.reduction.M);

                [hypothesesWeight, ~] = normalizeLogWeights(w_new);
                [~, I] = max(hypothesesWeight);
                state = multiHypotheses_new(I);

                estimates{i} = state.x;
                
                % for each hypothesis, do prediction
                for k = 1:length(w_new)
                    state_pred(k) = GaussianDensity.predict(multiHypotheses_new(k), motionmodel);    %prediction
                end
            end
        end   
    end
end

