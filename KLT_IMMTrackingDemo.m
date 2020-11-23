% ENGI 9805 Final Project Part 4
% Kanade-Lucas-Tomasi (Inverse Compositional Algorithm) Tracking with
% Interactive Multiple Model Kalman Filter
% Patrick Glavine
% Student#: 200901825

% clc
clear all
close all

% Set Kalman Filter Variance Parameters
% Process Variance: Control system trust in motion prediction
q1x = 0.001;
q1y = 0.001;
q2x = 0.001;
q2y = 0.001;
% Measurment Variance: Control system trust in measurements
r1x = 0.1;
r1y = 0.1;
r2x = 0.1;
r2y = 0.1;
% Set KLT Measurement Frequency, less measurements, less accuracy
num = 2;
% Number of iterations for KLT convergence (May need to be set high if
% there is large target motion between frames, optical flow may be too
% large. This can be solved by implementing pyramids when calculating
% optical flow.)
its = 40;
% Set movie 1-on 0-off for displaying results
showMovie = 1;
% Fast sets the increase playback frames per second of movie
fast = 1;
speed = 60; % FPS

% Select Test 0-No Test, 1-Circle Motion, 2-Linear Constant Velocity,
% 3-Linear Constant Acceleration
% If test = 0, type the video file link name below in the video reader
test = 0;

% Update target candidate over several frames, update = 1 turns on target
% updating, numframes is how often target is updated. Leave as zero if
% target profile does not change significantly during tracking.
update = 0;
numframes = 20;
check = numframes;

% Turn on tracking trail for center
trail = 0;
tindex = 1; % Trail index
fpmarker = 3; % Frames per marker in trail.

% Samples for averaging Kalman filter results, set to 1 if only running
% once.
samples = 1;

% Target lost flag
target_lost = 0;

for TEST = 1:samples
    
    % Read input video
    
    % Test 0 allows the user to input any video and track a selected target
    if test == 0
        movieObj = VideoReader('soccer.mov');
        get(movieObj);
        Width = movieObj.Width;
        Height = movieObj.Height;
        Frames = (movieObj.NumberOfFrames);
        images = read(movieObj);
        
        % Test 1 opens a circular motion tracking test with predefined starting position.
    elseif test == 1
        movieObj = VideoReader('CircleMotion1.avi');
        get(movieObj);
        Width = movieObj.Width;
        Height = movieObj.Height;
        Frames = (movieObj.NumberOfFrames);
        images = read(movieObj);
        
        % Test 2 opens a linear velocity tracking test with predefined starting position.
    elseif test == 2
        movieObj = VideoReader('LinearMotionCVTest.avi');
        get(movieObj);
        Width = movieObj.Width;
        Height = movieObj.Height;
        Frames = (movieObj.NumberOfFrames);
        images = read(movieObj);
        
        % Test 3 opens a linear acceleration tracking test with predefined starting position.
    elseif test == 3
        movieObj = VideoReader('LinearMotionAcc.avi');
        get(movieObj);
        Width = movieObj.Width;
        Height = movieObj.Height;
        Frames = (movieObj.NumberOfFrames);
        images = read(movieObj);
    end
    
    % Set all images to grayscale for KLT Processing
    for i = 1:Frames
        img(:,:,i) = double(rgb2gray(images(:,:,:,i)));
    end;
    
    %% Select initial Target Model
    
    if test == 0
        sprintf('Click and drag to draw box around target.')
        imshow(images(:,:,:,1))
        rect = round(getrect);
        x = rect(1);
        y = rect(2);
        w = rect(3); % Width of target window
        h = rect(4); % Height of target window
        
        % Starting window for circle test
    elseif test == 1
        x = 666;
        y = 300;
        w = 90;
        h = w;
        
        % Starting window for linear motion test
    else
        x = 98;
        y = 311;
        w = 67;
        h = w;
    end
    
    close all
    
    % Offsets for tracking center of target
    xoff = round(w/2);
    yoff = round(h/2);
    
    % Initialize filters for image gradients
    sobel_y=[ -1 -2 -1;0 0 0; 1 2 1];
    sobel_x=sobel_y';
    
    % Jacobian of warping parameters (Only translation assumed in tracker).
    dWdp = eye(2);
    
    % Create set of images for video output
    VideoOutput = images;
    
    % Initialize State Vectors for each model, 6x1 vectors used just to
    % simplify matrix calculations. First model only considers 4 states in
    % reality.
    X1(:,1) = [x y 0 0 0 0]';
    X2(:,1) = [x y 0 0 0 0]';
    
    % System update matrices
    F1 = [0 0 1 0; 0 0 0 1; 0 0 0 0; 0 0 0 0];
    F2 = [0 0 1 0 0 0; 0 0 0 1 0 0; 0 0 0 0 1 0; 0 0 0 0 0 1; 0 0 0 0 0 0; 0 0 0 0 0 0];
    
    % Noise matrices
    G1 = [0 0; 0 0; 1 0; 0 1];
    G2 = [0 0; 0 0; 0 0; 0 0; 1 0; 0 1];
    
    % Initialize combined update estimate and covariance matrix
    X_hatC =[x y 0 0 0 0]';
    %P_hatC = eye(6);
    
    % Initialize individual model estimates
    X_hat1 = zeros(6,Frames-1);
    X_hat2 = zeros(6,Frames-1);
    X_hat1(:,1) = [x y 0 0 0 0]';
    X_hat2(:,1) = [x y 0 0 0 0]';
    
    % Set measurement matrices
    H1 = [1 0 0 0; 0 1 0 0];
    H2 = [1 0 0 0 0 0; 0 1 0 0 0 0];
    
    % Initialize measurements for each filter
    Y1 = zeros(2,Frames-1);
    Y1(:,1) = H1*X1(1:4,1);
    Y2 = zeros(2,Frames-1);
    Y2(:,1) = H2*X2(:,1);
    
    % Set process variances for each model
    Q1 = diag([q1x q1y]);
    Q2 = diag([q2x q2y]);
    
    % Set Measurment variances for each model
    R1 = diag([r1x r1y]);
    R2 = diag([r2x r2y]);
    
    % Initialize system covariance matrix for each model
    P1 = eye(6);
    P2 = eye(6);
    
    % Create a state switching matrix, values are estimated here.
    p11 = 0.9;
    p12 = 1-p11;
    p22 = 0.9;
    p21 = 1-p22;
    p_ij = [p11 p12; p21 p22];
    
    % Create identity matrices for calculations
    I1 = eye(4);
    I2 = eye(6);
    
    % Initialize model probabilities
    MU = zeros(2,Frames-1);
    mu_i = [0.5 0.5];
    MU(:,1) = mu_i';
    
    % Set number of modes being considered
    modes = 2;
    dt = 1;
    
    % Tracked x and y coordinates for error measurment during testing
    x_tracked = zeros(1,Frames-1);
    y_tracked = zeros(1,Frames-1);
    x_tracked(1) = x+xoff;
    y_tracked(1) = y+yoff;
    meas = num; % Measurement frequency
    
    % Counts to see how often the constant acceleration or constant
    % velocity model probability is highest
    CA = 0;
    CV = 0;
    
    % Initialize trace variable used to create a trail behind tracked
    % object to illustrate trajectory.
    trace = {};
    trace{1} = [x y];
    
    % Draw initial position
    VideoOutput(:,:,:,1) = insertShape(images(:,:,:,1),'rectangle', [x,y,w,h], 'Color', 'red', 'LineWidth', 2);
    VideoOutput(:,:,:,1)=insertShape(images(:,:,:,1),'rectangle',[trace{1}(1,1)+xoff trace{1}(1,2)+yoff 2 2],'LineWidth',3,'Color','yellow');
    % Loop through every remaining frame in sequence
    for k = 1:Frames-1
        
        % Condition for checking if KLT should be applied during
        % current frame. Used to skip KLT process and allow Kalman
        % filter to track on its own between frames. This can speed up the
        % process but reduce accuracy in tracking.
        if mod(meas,num) == 0
            if k == 1
                % Obtain first target model to be used for remainder of tracking
                Target = img(y:y+h,x:x+w,k);
                % Calculate image x and y gradients
                grad_x=imfilter(Target,sobel_x);
                grad_y=imfilter(Target,sobel_y);
                [n,m] = size(Target);
                
                % Compute the Hessian of the target template
                Hessian = zeros(2,2);
                for i=1:n
                    for j = 1:m
                        Hessian(1,1) = Hessian(1,1) + (grad_x(i,j))^2;
                        Hessian(1,2) = Hessian(1,2) + (grad_x(i,j)*grad_y(i,j));
                        Hessian(2,2) = Hessian(2,2) + (grad_y(i,j))^2;
                    end
                end
                Hessian(2,1) = Hessian(1,2);
            end
            
            % This updates the target if the target is expected to change
            % shape during the tracking
            if update == 1
                if mod(check,numframes) == 0
                    Target = img(y:y+h,x:x+w,k);
                    % Calculate image x and y gradients
                    grad_x=imfilter(Target,sobel_x);
                    grad_y=imfilter(Target,sobel_y);
                    [n,m] = size(Target);
                    
                    % Compute the Hessian of the target template
                    Hessian = zeros(2,2);
                    for i=1:n
                        for j = 1:m
                            Hessian(1,1) = Hessian(1,1) + (grad_x(i,j))^2;
                            Hessian(1,2) = Hessian(1,2) + (grad_x(i,j)*grad_y(i,j));
                            Hessian(2,2) = Hessian(2,2) + (grad_y(i,j))^2;
                        end
                    end
                    Hessian(2,1) = Hessian(1,2);
                end
            end
            
            count = 0; % Used to limit number of iterations
            epsilon = 0.05; % Threshold for convergence
            
            % Initialize image warping parameters
            p = [0 0]';
            dp = [10,10]'; % Set dp high enough to initiate while loop
            
            while(norm(dp) > epsilon && (count < its))
                count = count + 1;
                
                % Check to see if the target has left the screen.
                if isnan(y+h+p(2)) || isnan(x+w+p(1)) || (y+h+p(2)) > Height || y <= 0 || (x+w+p(1)) > Width || x <= 0
                    sprintf('The target has been lost.')
                    target_lost = 1;
                    break
                end
                
                % Create grid for interpolating intensities of warped image.
                % Grid indices are shifted by current image warping
                % parameters and interpolation is performed to compute the
                % image intensities at the shifted grid locations.
                [x_grid, y_grid] = meshgrid(x+p(1):(x+p(1))+(w),y+p(2):(y+p(2))+(h));
                I_warped = interp2(img(:,:,k+1), x_grid, y_grid,'cubic');
                
                % Calculate the error image. Note that this is error image
                % is defined I_warped - Target in the referenced paper,
                % however the tracker was shifting the window in the
                % directions opposite to the directions followed by the
                % target. Calculating the image error this way fixed this
                % error.
                E = Target - I_warped;
                
                % Multiply gradients by the error image
                GEx = grad_x.*double(E);
                GEy = grad_y.*double(E);
                
                % Update x and y increments of image warping parameters
                dp = Hessian\[sum(GEx(:)),sum(GEy(:))]';
                
                % Update image warping parameters in x and y direction and
                % check to see if the norm of dp is less than theshold
                p(1) = p(1)+dp(1);
                p(2) = p(2)+dp(2);
            end
        end
        
        % Stop tracking if target has been lost
        if target_lost == 1
            break
        end
        
        % Update KLT Measurement
        x = (x+p(1));
        y = (y+p(2));
        
        % Calculate normalization vector to maintain model probability of 1.
        phibar_j = zeros(1,modes);
        for j = 1:modes
            for i = 1:modes
                phibar_j(j) = phibar_j(j) + p_ij(i,j)*mu_i(i);
            end
        end
        
        % Compute conditional model probabilities, ie: probabilities of switching
        % from one model to another or staying in current state.
        for i = 1:modes
            for j = 1:modes
                mu_ij(i,j) = (1/phibar_j(j))*p_ij(i,j)*mu_i(i);
            end
        end
        
        % Determine mixed state values for each model at increment k.
        X_hat0j1 = zeros(6,1);
        X_hat0j2 = X_hat0j1;
        for j = 1:modes
            for i = 1:modes
                if j==1 && i == 1
                    X_hat0j1(1:4,1) = X_hat0j1(1:4,1) + X_hat1(1:4,k)*mu_ij(i,j);
                elseif j==1 && i==2
                    X_hat0j1(:,1) = X_hat0j1(:,1) + X_hat2(:,k)*mu_ij(i,j);
                elseif j==2 && i==1
                    X_hat0j2(1:4,1) = X_hat0j2(1:4,1) + X_hat1(1:4,k)*mu_ij(i,j);
                else
                    X_hat0j2(:,1) = X_hat0j2(:,1) + X_hat2(:,k)*mu_ij(i,j);
                end
            end
        end
        
        % Compute mixed covariances for each model at time k.
        P_0j1 = zeros(6,6);
        P_0j2 = P_0j1;
        
        for j = 1:modes
            for i = 1:modes
                if j==1 && i == 1
                    P_0j1(1:4,1:4) = P_0j1(1:4,1:4)+ mu_ij(i,j)*(P1(1:4,1:4) + (X_hat1(1:4,k)-X_hat0j1(1:4,1))*(X_hat1(1:4,k)-X_hat0j1(1:4,1))');
                elseif j==1 && i==2
                    P_0j1 = P_0j1(:,:)+ mu_ij(i,j)*(P2 + (X_hat2(:,k)-X_hat0j1(:,1))*(X_hat2(:,k)-X_hat0j1(:,1))');
                elseif j==2 && i==1
                    P_0j2(1:4,1:4) = P_0j2(1:4,1:4)+ mu_ij(i,j)*(P1(1:4,1:4) + (X_hat1(1:4,k)-X_hat0j2(1:4,1))*(X_hat1(1:4,k)-X_hat0j2(1:4,1))');
                else
                    P_0j2 = P_0j2+ mu_ij(i,j)*(P2 + (X_hat2(:,k)-X_hat0j2(:,1))*(X_hat2(:,k)-X_hat0j2(:,1))');
                end
            end
        end
        
        % Update the 1st Kalman Filter Prediction Model
        X1(:,k+1) = [x y 0 0 0 0]';
        
        % Model 1 measurement update
        Y1(:,k+1) = H1*X1(1:4,k+1) + (sqrt(R1)*rand(2,1));
        
        % Model 1 estimatino update
        X_hat1(1:4,k+1) = X_hat0j1(1:4,1) + (F1*X_hat0j1(1:4,1))*dt;
        
        % Model 1 covariance matrix update
        P1 = (I1+F1*dt)*P_0j1(1:4,1:4)*(I1+F1*dt)' + G1*Q1*G1'*dt^2;
        S1 = H1*P1*H1'+R1;
        K1 = P1*H1'*(S1^-1);
        
        % Model 1 innovation
        Z1 = K1*(Y1(:,k+1)- H1*X_hat1(1:4,k+1));
        
        % Model 1 correction step
        X_hat1(1:4,k+1) = X_hat1(1:4,k+1) + Z1;
        
        % Model 1 covariance matrix correction
        P1 = P1 - K1*H1*P1;
        
        % Update the 2nd Kalman Filter Prediction Model
        X2(:,k+1) = [x y 0 0 0 0]';
        
        % Model 2 measurement update
        Y2(:,k+1) = H2*X2(:,k+1) + (sqrt(R2)*rand(2,1));
        
        % Model 2 prediction
        X_hat2(:,k+1) = X_hat0j2(:,1) + (F2*X_hat0j2(:,1))*dt;
        
        % Model 2 covariance matrix update
        P2 = (I2+F2*dt)*P_0j2*(I2+F2*dt)' + G2*Q2*G2'*dt^2;
        S2 = H2*P2*H2'+R2;
        K2 = P2*H2'*(S2^-1); % Kalman Gain
        
        % Model 2 innovation
        Z2 = K2*(Y2(:,k+1) - H2*X_hat2(:,k+1));
        
        % Model 2 correction step
        X_hat2(:,k+1) = X_hat2(:,k+1) + Z2;
        
        % Model 2 covariance matrix coreection
        P2 = P2 - K2*H2*P2;
        
        % Compute likelihood of each model
        if det(2*pi*S1)>0
            lambda1 = (1/(sqrt(det(2*pi*S1))))*exp((-0.5*((Z1(1:2,1))')*((S1^-1)*(Z1(1:2,1)))));
        end
        
        if det(2*pi*S2)> 0
            lambda2 = (1/(sqrt(det(2*pi*S2))))*exp((-0.5*((Z2(1:2,1))')*((S2^-1)*((Z2(1:2,1))))));
        end
        if lambda1 == 0 && lambda2==0
            lambda1 = 0.1;
            lambda2 = 0.1;
        end
        if lambda1 > 0 && lambda2==0
            lambda1 = 1;
            lambda2 = 0;
        end
        if lambda2 > 0 && lambda1==0
            lambda1 = 0;
            lambda2 = 1;
        end
        
        % Compute normalization constant
        c = lambda1*phibar_j(1) + lambda2*phibar_j(2);
        
        % Update the probability of each model
        mu_i(1) = (1/c)*lambda1*phibar_j(1);
        mu_i(2) = (1/c)*lambda2*phibar_j(2);
        MU(:,k+1) = [mu_i(1) mu_i(2)]';
        
        % Compute the combined state estimate from the two filters
        X_hatC(1:4,k+1) = X_hat1(1:4,k+1)*mu_i(1);
        X_hatC(:,k+1) = X_hatC(:,k+1) + X_hat2(:,k+1)*mu_i(2);
        X_hatC = abs(X_hatC);
        % Compute the combined covariance from the two filters
        % P_hatC(1:4,1:4) = mu_i(1)*(P1(1:4,1:4)+(X_hat1(1:4,k+1)-X_hatC(1:4,k+1))*(X_hat1(1:4,k+1)-X_hatC(1:4,1))');
        % P_hatC = P_hatC + mu_i(2)*(P2 +(X_hat2(:,1)-X_hatC(:,1))*(X_hat2(:,1)-X_hatC(:,1))');
        
        % Update the predicted position of the target.
        x = (X_hatC(1,k+1));
        y = (X_hatC(2,k+1));
        
        % Round result since pixels are discrete integer values
        x = round(x);
        y = round(y);
        
        % Make sure x and y are numbers in case of matrix singularity
        if isnan(x) || isnan(y)
            x = round(x_tracked(k)-w/2)+1;
            y = round(y_tracked(k)-h/2)+1;
        end
        
        % Log tracked position
        x_tracked(k+1) = round((x + w/2));
        y_tracked(k+1) = round((y + h/2));
        
        % Add a marker every few frames depending on fpmarker value
        if mod(k,3) == 0
            tindex = tindex+1;
            trace{tindex} = [x y];
        end
        
        % Draw object trailing markers if turned on
        if trail ==1
            for i = 1:tindex
                if i ==tindex
                    VideoOutput(:,:,:,k+1)=insertShape(images(:,:,:,k+1),'rectangle',[trace{i}(1,1)+xoff trace{i}(1,2)+yoff 2 2],'LineWidth',3,'Color','yellow');
                else
                    VideoOutput(:,:,:,k+1)=insertShape(images(:,:,:,k+1),'rectangle',[trace{i}(1,1)+xoff trace{i}(1,2)+yoff 2 2],'LineWidth',3,'Color','green');
                end
            end
        end
        
        % Draw rectangle around tracked target for visualization.
        % Output color corresponding to model with current highest
        % probability.
        if mu_i(1) > mu_i(2)
            CV = CV + 1;
            VideoOutput(:,:,:,k+1)=insertShape(images(:,:,:,k+1),'rectangle',[x y w h],'LineWidth',3,'Color','red');
        else
            CA = CA + 1;
            VideoOutput(:,:,:,k+1)=insertShape(images(:,:,:,k+1),'rectangle',[x y w h],'LineWidth',3,'Color','yellow');
        end
        
        % Used for taking KLT measurement at particular frequency.
        meas = meas+1;
        if meas == 2*num
            meas = num;
        end
        
        % Used to increment target update count
        check = check+1;
        if check == 2*numframes
            check = numframes;
        end
    end
    
    %% Display results as a movie
    
    if showMovie == 1
        for i = 1:Frames
            % Resize image if it does not fit in figure window.
            if Width > 420 || Height > 560
                if Width > Height
                    movie_frame(:,:,:,i) = imresize(VideoOutput(:,:,:,i),[550 750]);
                else
                    movie_frame(:,:,:,i) = imresize(VideoOutput(:,:,:,i),[750 550]);
                end
                M(i) = im2frame(movie_frame(:,:,:,i));
            else
                M(i) = im2frame(VideoOutput(:,:,:,i));
            end
        end
        %%
        figure(3)
        title('Tracking Results')
        if Width > 420 || Height > 560
            axis([0 0.5 0 0.8])
        end
        % Display movie with high fps rate if fast is equal to 1
        if fast == 1
            movie(M,3,speed)
        else
            movie(M)
        end
    end
    
    
    %% Error Measurement for Testing purposes.
    if test == 1
        % Create Ellipse Coordinates for Comparison during circle test,
        % ellipse used because circle became skewed during getframe
        % process.
        
        a=277; % Major and minor axes of ellipse
        b=218;
        xi=435; % Ellipse center
        yi=345;
        th=0:pi/200:2*pi;
        
        % Generate ellipse points.
        xe=xi+a*cos(th);
        ye=yi-b*sin(th);
        
        x_actual = xe';
        y_actual = ye';
        x_tracked = x_tracked';
        y_tracked = y_tracked';
        
        % Calculate error between actual object locations and tracked
        % values.
        errorx = x_actual-x_tracked;
        errory = y_actual-y_tracked;
        error_meanx(TEST) = sum(abs(errorx(:)))/Frames
        error_meany(TEST) = sum(abs(errory(:)))/Frames
        
    elseif test == 2
        % Linear velocity Coordinates for Comparison
        x_start = 132;
        y_start = 344;
        x_end = 737;
        x_pixels(1) = x_start;
        y_end = y_start;
        
        % Generate a set of pixel locations from linear velocity video
        for i = 2:165
            x_pixels(i) = x_pixels(i-1) + (x_end-x_start)/165;
        end
        x_actual = x_pixels';
        y_actual = (y_start*ones(1,length(x_pixels)))';
        x_tracked = x_tracked';
        y_tracked = y_tracked';
        
        % Calculate error between actual object locations and tracked
        % values.
        errorx = x_actual-x_tracked;
        errory = y_actual-y_tracked;
        error_meanx = sum(abs(errorx(:)))/Frames
        error_meany = sum(abs(errory(:)))/Frames
        
    elseif test == 3
        % Linear acceleration Coordinates for Comparison
        
        % Calculate actual plot coordinates used during acceleration video
        % production.
        for i = 1:55
            xh(i) = 3+0.005*i^2;
        end
        x_start = 132;
        x_s = [132 158 227 340 496 632 718 788];
        y_start = 344;
        x_end = 788;
        y_end = y_start;
        
        % Calculate approximate acceleration values using pixel samples
        % from video.
        acc = (x_s(1)/(xh(1))+x_s(2)/(xh(11))+x_s(3)/(xh(21))+x_s(4)/(xh(31))+x_s(5)/(xh(41))+...
            x_s(6)/(xh(48))+x_s(7)/(xh(52))+x_s(8)/(xh(55)))/8;
        
        % Generate acceleration data
        x_pixels = (xh*acc);
        x_actual = x_pixels';
        y_actual = (y_start*ones(1,length(x_pixels)))';
        x_tracked = x_tracked';
        y_tracked = y_tracked';
        
        % Calculate error between actual object locations and tracked
        % values.
        errorx = x_actual-x_tracked;
        errory = y_actual-y_tracked;
        error_meanx = sum(abs(errorx(:)))/Frames;
        error_meany = sum(abs(errory(:)))/Frames;
        
    end
end

if test == 0
else
    % Calculate mean pixel error over set number of trials.
    x_error_mean_Total = sum(error_meanx(:))/samples;
    y_error_mean_Total = sum(error_meany(:))/samples;
end

% Plot model probabilities to see contributions during tracking
figure(4)
plot(MU(1,:))
title('Constant Velocity Model Probability')
figure(5)
plot(MU(2,:))
title('Constant Acceleration Model Probability')