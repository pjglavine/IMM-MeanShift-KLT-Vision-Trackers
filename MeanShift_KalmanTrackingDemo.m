% ENGI 9805 Final Project Part 1
% Mean Shift Object Tracking with Constant Velocity and Constant 
% Acceleration Kalman Filters
% Patrick Glavine
% Student # 200901825

% clc
clear all
close all

% Set Kalman Filter 0-No Filter, 1-Constant Velocity Kalman, 2-Constant
% Acceleration Kalman
Kalman = 2;

% Set Kalman Filter Variance Parameters
% Process Variance: Control system trust in motion prediction
sigmaQ1 = 0.001; % Process variance
sigmaQ2 = 0.001;

% Measurment Variance: Control system trust in measurements
sigmaR1 = 0.1; % Measurement variance
sigmaR2 = 0.1;

% Set Mean Shift Measurement Frequency and iterations used for convergence
num = 1;
its = 20;

% Set movie for displaying results
showMovie = 1;
fast = 1; % Sets movie playback to play faster than default speed
speed = 60; % FPS when fast set to 1

% Select Test 0-No Test, 1-Circle Motion, 2-Linear Constant Velocity,
% 3-Linear Constant Acceleration
% If test = 0, type the video file link name below in the video reader
test = 0;

% Turn on tracking trail for center 1-on
trail = 0;
tindex = 1; % Number of frames for trail update
fpmarker = 2; % Frames per marker in trail.

% Samples for averaging Kalman filter results, set to 1 if only running
% once.
samples = 1;

% Target lost flag
target_lost = 0;

for TEST = 1:samples
    
    % Read input video
    
    % Test 0 allows the user to input any video and track a selected target
    if test == 0
        movieObj = VideoReader('juggling.mov');
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
    
    % Create Normalized Gaussian Kernel window for target probability density function
    kernel = fspecial('gaussian',[h,w],((h+w)/12));
    
    % Calculate x and y gradient of image
    [gx,gy] = imgradientxy(-kernel);
    
    % Calculate the gradient magnitude for each point within the kernel
    % window
    gradmag = zeros(size(kernel));
    for i = 1:h
        for j = 1:w
            gradmag(i,j) = sqrt((gx(i,j)^2)+(gy(i,j)^2));
        end
    end
    
    % Create indexed color map for first image.
    [Target,full_color_map] = rgb2ind(images(:,:,:,1),65536);
    
    % Use indexed image with specified target window.
    Target = Target(y:y+h,x:x+w);
    
    % Offset indexed image by 1. Index values of the target
    % image are used as histogram bin values. This offset ensures that the
    % color black does not produce a value of 0 which causes an error.
    Target=Target(:,:)+1;
    
    %Calculate color histogram for target frame using colors found in
    %indexed image.
    q = zeros(length(full_color_map),1);
    for i = 1:h
        for j = 1:w
            bin = Target(i,j); % Set bin value to pixel index number
            q(bin) = q(bin) + kernel(i,j); % Add kernel weight to pdf
        end
    end
    
    % Initialize Constant Velocity Kalman Filter
    F1 = [0 0 1 0; % System Matrix
        0 0 0 1;
        0 0 0 0;
        0 0 0 0];
    
    Q1 = [sigmaQ1 0; 0 sigmaQ2]; % Process Variance Matrix
    H1 = [1 0 0 0; 0 1 0 0]; % Measurement Matrix
    R1 = [sigmaR1 0; 0 sigmaR2];% Measurement Variance Matrix
    G1 = [0 0;0 0; 1 0; 0 1]; % Process Noise Matrix
    P1 = eye(4);  % Process Covariance Matrix
    I1 = P1; % Identity matrix
    
    % Measurements
    Y1 = zeros(2,Frames-1);
    Y1(:,1) = H1*[x y 0 0]';
    
    % State Estimates
    X1 = zeros(4,Frames-1);
    X_hat1 = zeros(4,Frames-1);
    X_hat1(:,1) = [x,y,0,0]';
    
    % Initialize Constant Acceleration Kalman Filter
    F2 = [0 0 1 0 0 0;
        0 0 0 1 0 0;
        0 0 0 0 1 0;
        0 0 0 0 0 1;
        0 0 0 0 0 0;
        0 0 0 0 0 0];
    
    Q2 = [sigmaQ1 0; 0 sigmaQ2];
    H2 = [1 0 0 0 0 0; 0 1 0 0 0 0];
    R2 = [sigmaR1 0; 0 sigmaR2];
    G2 = [0 0;0 0; 0 0; 0 0; 1 0; 0 1];
    P2 = eye(6);
    I2 = P2;
    
    dt = 1; % System update frequency i.e.: 1 frame per filter update
    
    % Measurements
    Y2 = zeros(2,Frames-1);
    Y2(:,1) = H2*[x y 0 0 0 0]';
    
    % State Estimates
    X2 = zeros(6,Frames-1);
    X_hat2 = zeros(6,Frames-1);
    X_hat2(:,1) = [x,y,0,0,0 0]';
    
    % Tracked x and y coordinates for error measurment during testing
    x_tracked = zeros(1,Frames-1);
    y_tracked = zeros(1,Frames-1);
    x_tracked(1) = x+xoff;
    y_tracked(1) = y+yoff;
    
    meas = num; % Measurement frequency variable
    
    % Initialize trace variable used to create a trail behind tracked
    % object to illustrate trajectory.
    trace = {};
    trace{1} = [x y];
    
    % Draw initial position
    images(:,:,:,1)=insertShape(images(:,:,:,1),'rectangle',[trace{1}(1,1)+xoff trace{1}(1,2)+yoff 2 2],'LineWidth',3,'Color','yellow');
    images(:,:,:,1)=insertShape(images(:,:,:,1),'rectangle',[x y w h],'LineWidth',2,'Color','blue');
    
    % Mean shift convergence threshold
    epsilon = 0.001;
    
    % For all remaining frames in sequence.
    for k=1:Frames-1
        
        % Condition for checking if mean shift should be applied during
        % current frame. Used to skip mean shift process and allow Kalman
        % filter to track on its own between frames. This can speed up the
        % process but reduce accuracy in tracking.
        if mod(meas,num) == 0
            
            % Obtain color map of next frame in sequence.
            Next_Frame = rgb2ind(images(:,:,:,k+1),full_color_map);
            
            % Offset indices of indexed image.
            Next_Frame = Next_Frame(:,:) + 1;
                            
            % Make sure that target window is not going off the screen.
            if isnan(y+h) || isnan(x+w) || (y+h) > Height || y <= 0 || (x+w) > Width || x <= 0
                sprintf('The target has been lost.')
                target_lost = 1;
                break
            end
            
            % Select new target window.
            Target = Next_Frame(y:y+h,x:x+w);
            
            % Calculate color histogram for target candidate in the current frame
            p = zeros(size(full_color_map,1),1);
            for i = 1:h
                for j = 1:w
                    bin = Target(i,j); % Black gives 0 index value
                    p(bin) = p(bin) + kernel(i,j);
                end
            end
            
            % Assign weights to each pixel in current target candidate
            % window. Weights are based on a square root ratio of original
            % target and target candidate pdf values with the same color as
            % the current pixel evaluated.
            weights = zeros(size(kernel));
            for i = 1:h
                for j = 1:w
                    bin = Target(i,j);
                    if p(bin) > 0
                        weights(i,j) = sqrt(q(bin)/p(bin));
                    end
                end
            end
            % Normalize the weights
            weightsum = sum(weights(:));
            weights = weights./weightsum;
            
            % Similarity function distance used for convergence
            d = 1;
            
            count = 1; % Number of iterations allowed

            while ((d > epsilon) && (count < its))
                count = count + 1;
                
                % Initialize the mean shift vector
                MS = [0 0];
                weightsum = 0;
                
                % Mid point values used to calculate pixel distance from center
                % coordinate in target window.
                midx = round((h)/2);
                midy = round((w)/2);
                for i = 1:h
                    for j = 1:w
                        % Calculate mean shift vector in x direction
                        MS(1,1) = MS(1,1) + weights(i,j)*gx(i,j)*(midx-(abs(midx-i))); 
                        % Calculate mean shift vector in y direction
                        MS(1,2) = MS(1,2) + weights(i,j)*gy(i,j)*(midy-(abs(midy-j)));
                        % Calculate the sum of weighted gradients in the window
                        weightsum = weightsum + weights(i,j)*gradmag(i,j);
                    end
                end
                % Divide x and y components of mean shift vector by
                % weighted gradient magnitudes
                MS = MS./weightsum;
                
                % Apply mean shift to current position estimate
                x = round(x+MS(1,1));
                y = round(y+MS(1,2));
                
                % Make sure that target window is not going off the screen.
                if isnan(y+h) || isnan(x+w) || (y+h) > Height || y <= 0 || (x+w) > Width || x <= 0
                    sprintf('The target has been lost.')
                    target_lost = 1;
                    break
                end
                
                % Update target candidate colour histogram and weights
                Target = Next_Frame(y:y+h,x:x+w);
                p = zeros(length(full_color_map),1);
                for i = 1:h
                    for j = 1:w
                        bin = Target(i,j);
                        p(bin) = p(bin) + kernel(i,j);
                    end
                end
                
                weights = zeros(size(kernel));
                for i = 1:h
                    for j = 1:w
                        bin = Target(i,j);
                        if p(bin) > 0
                            weights(i,j) = sqrt(q(bin)/p(bin));
                        end
                    end
                end
                weightsum = sum(sum(weights(:)));
                weights = weights./weightsum;
                
                % Calculate Similarity using Bhattacharyya Coefficient
                l = size(p,1);
                BC = 0;
                for i = 1:l
                    rho = sqrt(p(i)*q(i));
                    BC = BC + rho;
                end
                
                % Check distance function and compare with theshold value
                d = sqrt(1-BC);
                
            end
        end
        
        % Stop tracking if target has been lost.
        if target_lost == 1
            break
        end
        
        if Kalman == 1
            % Update the Kalman Filter Prediction
            X1(:,k+1) = [x y 0 0]';
            
            % Measurement update
            Y1(:,k+1) = H1*X1(:,k+1) + (sqrt(R1)*rand(2,1));
            
            % Update state prediction
            X_hat1(:,k+1) = X_hat1(:,k) + (F1*X_hat1(:,k))*dt;
            
            % Covariance Matrix update
            P1 = (I1+F1*dt)*P1*(I1+F1*dt)' + G1*Q1*G1'*dt^2;
            S1 = H1*P1*H1'+R1;
            K1 = P1*H1'*(S1^-1);
            
            % Apply Kalman Filter Correction Step
            Innov1 = K1*(Y1(:,k+1)- H1*X_hat1(:,k+1));
            X_hat1(:,k+1) = X_hat1(:,k+1) + Innov1;
            P1 = P1 - K1*H1*P1;
            
            % Update estimate of target location in next frame
            x = (X_hat1(1,k+1));
            y = (X_hat1(2,k+1));
            
        elseif Kalman == 2
            
            % Use mean shift estimate to update Kalman filter estimate
            X2(:,k+1) = [x y 0 0 0 0]';
            
            % Measurement update
            Y2(:,k+1) = H2*X2(:,k+1) + (sqrt(R2)*rand(2,1));
            
            % Update state prediction
            X_hat2(:,k+1) = X_hat2(:,k) + (F2*X_hat2(:,k))*dt;
            
            % State Covariance update
            P2 = (I2+F2*dt)*P2*(I2+F2*dt)' + G2*Q2*G2'*dt^2;
            S2 = H2*P2*H2'+R2;
            K2 = P2*H2'*(S2^-1);
            
            % Correction step using measurement
            Innov2 = K2*(Y2(:,k+1)- H2*X_hat2(:,k+1));
            X_hat2(:,k+1) = X_hat2(:,k+1) + Innov2;
            P2 = P2 - K2*H2*P2;
            
            % Update estimate of target location in next frame
            x = (X_hat2(1,k+1));
            y = (X_hat2(2,k+1));
        end
        
        % Round result since pixels are discrete integer values
        x = round(x);
        y = round(y);
        
        % Log the tracked position
        x_tracked(k+1) = round((x + w/2));
        y_tracked(k+1) = round((y + h/2));
        
        % Add a marker every few frames depending on fpmarker value
        if mod(k,fpmarker) == 0
            tindex = tindex+1;
            trace{tindex} = [x y];
        end
        
        % Draw object trailing markers if turned on
        if trail ==1
            for i = 1:tindex
                if i ==tindex
                    images(:,:,:,k+1)=insertShape(images(:,:,:,k+1),'rectangle',[trace{i}(1,1)+xoff trace{i}(1,2)+yoff 2 2],'LineWidth',3,'Color','yellow');
                else
                    images(:,:,:,k+1)=insertShape(images(:,:,:,k+1),'rectangle',[trace{i}(1,1)+xoff trace{i}(1,2)+yoff 2 2],'LineWidth',3,'Color','green');
                end
            end
        end
        
        % Draw rectangle around tracked target for visualization.
        images(:,:,:,k+1)=insertShape(images(:,:,:,k+1),'rectangle',[x y w h],'LineWidth',3,'Color','blue');
        
        % Used for taking mean shift measurement at particular frequency.
        meas = meas+1;
        if meas == 2*num
            meas = num;
        end
    end
    
    %% Display results as a movie
    
    if showMovie == 1
        for i = 1:Frames
            % Resize image if it does not fit in figure window.
            if Width > 420 || Height > 560
                if Width > Height
                    movie_frame(:,:,:,i) = imresize(images(:,:,:,i),[550 750]);
                else
                    movie_frame(:,:,:,i) = imresize(images(:,:,:,i),[750 550]);
                end
                M(i) = im2frame(movie_frame(:,:,:,i));
            else
                M(i) = im2frame(images(:,:,:,i));
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
        error_meanx = sum(abs(errorx(:)))/Frames
        error_meany = sum(abs(errory(:)))/Frames
        
    end
end

if test == 0
else
    % Calculate mean pixel error over set number of trials.
    x_error_mean_Total = sum(error_meanx(:))/samples;
    y_error_mean_Total = sum(error_meany(:))/samples;
end