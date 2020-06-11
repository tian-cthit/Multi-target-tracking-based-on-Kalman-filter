classdef PHDfilter
    %PHDFILTER is a class containing necessary functions to implement the
    %PHD filter 
    %Model structures need to be called:
    %    sensormodel: a structure which specifies the sensor parameters
    %           P_D: object detection probability --- scalar
    %           lambda_c: average number of clutter measurements per time scan, 
    %                     Poisson distributed --- scalar
    %           pdf_c: value of clutter pdf --- scalar
    %           intensity_c: Poisson clutter intensity --- scalar
    %       motionmodel: a structure which specifies the motion model parameters
    %           d: object state dimension --- scalar
    %           F: function handle return transition/Jacobian matrix
    %           f: function handle return predicted object state
    %           Q: motion noise covariance matrix
    %       measmodel: a structure which specifies the measurement model parameters
    %           d: measurement dimension --- scalar
    %           H: function handle return transition/Jacobian matrix
    %           h: function handle return the observation of the target state
    %           R: measurement noise covariance matrix
    %       birthmodel: a structure array which specifies the birth model (Gaussian
    %       mixture density) parameters --- (1 x number of birth components)
    %           w: weights of mixture components (in logarithm domain)
    %           x: mean of mixture components
    %           P: covariance of mixture components
    
    properties
        density %density class handle
        paras   %parameters specify a PPP
    end
    
    methods
        function obj = initialize(obj,density_class_handle,birthmodel)
            %INITIATOR initializes PHDfilter class
            %INPUT: density_class_handle: density class handle
            %OUTPUT:obj.density: density class handle
            %       obj.paras.w: weights of mixture components --- vector
            %                    of size (number of mixture components x 1)
            %       obj.paras.states: parameters of mixture components ---
            %                    struct array of size (number of mixture
            %                    components x 1) 
            
            obj.density = density_class_handle;
            obj.paras.w = [birthmodel.w]';
            obj.paras.states = rmfield(birthmodel,'w')';
        end
        
        function obj = predict(obj,motionmodel,P_S,birthmodel)
            %PREDICT performs PPP prediction step
            %INPUT: P_S: object survival probability
            H = numel(obj.paras.w);
            birth_states = rmfield(birthmodel,'w')';
            
            % surviving objects
            obj.paras.w = obj.paras.w + log(P_S);
            for i = 1:H
                obj.paras.states(i) = GaussianDensity.predict(obj.paras.states(i), motionmodel); 
            end
            
            % birth process
            obj.paras.w = [obj.paras.w; [birthmodel.w]'];
            for i = 1 : numel(birthmodel)
                obj.paras.states(i+H) = birth_states(i);
            end
           
        end
        
        function obj = update(obj,z,measmodel,intensity_c,P_D,gating)
            %UPDATE performs PPP update step and PPP approximation
            %INPUT: z: measurements --- matrix of size (measurement dimension 
            %          x number of measurements)
            %       intensity_c: Poisson clutter intensity --- scalar
            %       P_D: object detection probability --- scalar
            %       gating: a struct with two fields: P_G, size, used to
            %               specify the gating parameters
            
            % undetected objects weights:
            lambda_PPP_w = log(1 - P_D) + obj.paras.w;
            
            % gating:
            H = numel(obj.paras.w);  % number of hypotheses(Gaussian mixtures)
            m = size(z, 2);
            z_ingate = cell(H, 1);
            meas_in_gate_bool = zeros(H, m);
            
            for h = 1:H
                [z_ingate{h}, meas_in_gate_bool(h, :)] = obj.density.ellipsoidalGating(obj.paras.states(h), z, measmodel, gating.size);
            end
            
            new_H = sum(sum(double(meas_in_gate_bool)));  % number of new hypotheses
            w_update = zeros(H + new_H, 1);
            w_update(1:H) = lambda_PPP_w;
            
            update = cell(H, m);    % an H x m cell, stores the updated states
            % detected objects update:
            w = zeros(H, m);
            counter = H;
            for i = 1:m
                for h = 1:H
                    if meas_in_gate_bool(h, i) == 1
                        update{h, i} = obj.density.update(obj.paras.states(h), z(:, i), measmodel);
                        obj.paras.states = [obj.paras.states; update{h, i}];
                        predicted_likelihood_log = obj.density.predictedLikelihood(obj.paras.states(h),z(:, i),measmodel);
                        w(h, i) = log(P_D) + obj.paras.w(h) + predicted_likelihood_log;
                        w(h, i) = exp(w(h, i));
                    end
                end
               
                w_sum = sum(w(:, i));
                
                for h = 1:H
                    if meas_in_gate_bool(h, i) == 1
                        counter = counter + 1;
                        w(h, i) = log( w(h, i)/(intensity_c + w_sum) );
                        w_update(counter) = w(h, i);
                    end
                end
                    
            end
            obj.paras.w = w_update;
            
        end
        
        function obj = componentReduction(obj,reduction)
            %COMPONENTREDUCTION approximates the PPP by representing its
            %intensity with fewer parameters
            
            %Pruning
            [obj.paras.w, obj.paras.states] = hypothesisReduction.prune(obj.paras.w, obj.paras.states, reduction.w_min);
            %Merging
            if length(obj.paras.w) > 1
                [obj.paras.w, obj.paras.states] = hypothesisReduction.merge(obj.paras.w, obj.paras.states, reduction.merging_threshold, obj.density);
            end
            %Capping
            [obj.paras.w, obj.paras.states] = hypothesisReduction.cap(obj.paras.w, obj.paras.states, reduction.M);
        end
        
        function estimates = PHD_estimator(obj)
            %PHD_ESTIMATOR performs object state estimation in the GMPHD filter
            %OUTPUT:estimates: estimated object states in matrix form of
            %                  size (object state dimension) x (number of
            %                  objects) 
            estimates = [];
            E = round(sum(exp(obj.paras.w)));    % expected number of objects.
            E = min(E, numel(obj.paras.states));
            
            [~, index] = maxk(obj.paras.w, E);
            for i = 1:E
                estimates = [estimates, obj.paras.states(index(i)).x];
            end
            

        end
        
    end
    
end

