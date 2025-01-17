%% MATLAB Automatic Differentiation Node
% complex and multidimention added by herve hugonnet c.f complex chain rule
% https://math.stackexchange.com/questions/1445815/complex-chain-rule-for-complex-valued-functions
% modify it to ease integration :
%                            f=f(z)   g=g(w)   h=g(f(z))
%      dh /dz =     dg/dw *      df /dz +conj(dconj(g)/dw)*(dconj(f)/dz)
%(dconj(h)/dz)=conj(dg/dw)*(dconj(f)/dz)+     dconj(g)/dw *      df /dz
% we compute df/dz'
%
%  Object-oriented reverse mode automatic differentiation.
%  Modified for use here, originally available (04/2017) from:
%   https://github.com/gaika/madiff
%
%  LICENSE NOTICE
%  This file is distributed under the GNU General Public License (GPL)
%  (See ADNode_LICENSE.txt)
%
%  The GNU GPL allows redistribution of the original program and modified
%  versions thereof. In accordance with term 2.a of the GNU GPL this notice
%  states the modifications made to the original program:
%
%   March 2017
%   - Added overload for the four-quadrant arctangent function:
%      atan2, atan_backprop
%   - Modified overload for mtimes to allow for non-square matrices:
%      mtimes
%   - Added overload for left matrix divide (A\b):
%      mldivide, mldivide_backprop
%   - Added overload for transpose operator (A'):
%      ctranspose
%
%

classdef ADNode < handle
    %% Node in the function evalution graph
    properties (Constant)
        real_optimization=true;
    end
    properties
        value % function value at this node
        grad % gradient accumulator %conj(df)/dx
        grad_c % gradient accumulator %df/d(x)
        func % callback function to update gradient of the parent nodes
        root % input node that holds the tape
        tape % sequence of evaluation steps
        reuse=false;% set to true to make all variable reusable might use more memory
        real_valued=false;
        sz_root=[];
        %debug_var;
    end

    methods
        function y = ADNode(x, root, func)
            %% create new node
            if nargin > 1;
                y.func = func;
                y.root = root;
                root.tape{end+1} = y;
            else
               display('ADNode warning :might steel be able to minimize computation of grad_c for abs2 ?');
                y.sz_root=size(x);
                y.root = y;
                y.tape = {};
            end
            if isempty(x)
                error(['Node value is empty might have been deleted in another operation.' ...
                    'If reusing a node set the reuse parameter to true using the reusable function. ' ...
                    'Or can just put the default property to true (is not so bad for performance if not using lot of memory). ' ...
                    'ex : y=reusable(x); z=y+log(y);']);
            end
            y.value = x;
        end
        function [gradient] = backprop(x, dy,dy_c)
            %% backpropagate the gradient by evaluating the tape backwards
            if nargin > 1
                x.grad = dy;
                x.grad_c = dy_c;
            else
                x.grad = 1;
                x.grad_c = []; %x we compute the derivative of x
            end

            if ~x.root.tape{end}.real_valued && x.root.real_optimization
                error(['The cost function is not real valued might not work for gradient descent. ' ...
                    'If you want to proceed anyway set the constant property of the ADNode : real_optimization to false at the root. ' ...
                    'Else add a real() abs() abs2() or so on at the end of your cost function.']);
            end

            for k = length(x.root.tape):-1:1
                %{
                display('-- eval --');
                display(size(x.root.tape{k}.value));
                display(x.root.tape{k}.func);
                display(x.root.tape{k}.real_valued);
                %}
                x.root.tape{k}.func(x.root.tape{k});
                %display(isreal(x.root.tape{k}.grad));
                %display(isreal(x.root.tape{k}.grad_c));
                x.root.tape(k) = [];
            end
            gradient = x.root.grad;
            %x.root.sz_root
            %size(gradient)
            if ~isempty(gradient)
                
                gradient=rep_to_good_size(gradient,x.root.sz_root);
                for ii=1:length(size(gradient))
                    if x.root.sz_root(ii)~=size(gradient, ii) && x.root.sz_root(ii)==1
                        gradient=sum(gradient,ii);
                    end
                end
            end
            gradient=conj(gradient);

        end

        %padd_crop_to_fit(img_out,sizing)
        
        function y = padd_crop_to_fit(x,sizing)
            sz=size(x.value);
            y = ADNode(padd_crop_to_fit(x.value,sizing), x.root, @(y) x.add((padd_crop_to_fit(y.grad,sz)),(padd_crop_to_fit(y.grad_c,sz))));
            %y.debug_var=x.func;
            if x.real_valued
                y.real_valued=true;
            end
            if ~x.reuse
                x.value=[];%save memory
            end
        end
        
        function y = reshape(x,sizing)
            sz=size(x.value);
            y = ADNode(reshape(x.value,sizing), x.root, @(y) x.add((reshape_if_not_empty(y.grad,sz)),(reshape_if_not_empty(y.grad_c,sz))));
            if x.real_valued
                y.real_valued=true;
            end
            if ~x.reuse
                x.value=[];%save memory
            end
        end
        
        function y=radon_adiff(x,angles,over_padd,use_filter)
            y = ADNode(radon_adiff(x.value,angles,over_padd,use_filter), x.root, @(y) x.add(iradon_adiff(y.grad,angles,over_padd,use_filter),iradon_adiff(y.grad_c,angles,over_padd,use_filter)));
            %y.debug_var=x.func;
            if x.real_valued
                y.real_valued=true;
            end
            if ~x.reuse
                x.value=[];%save memory
            end
        end
        function y=iradon_adiff(x,angles,over_padd,use_filter)
            y = ADNode(iradon_adiff(x.value,angles,over_padd,use_filter), x.root, @(y) x.add(radon_adiff(y.grad,angles,over_padd,use_filter),radon_adiff(y.grad_c,angles,over_padd,use_filter)));
            %y.debug_var=x.func;
            if x.real_valued
                y.real_valued=true;
            end
            if ~x.reuse
                x.value=[];%save memory
            end
        end
        
        function y = permute(x,dimorder)
            y = ADNode(permute(x.value,dimorder), x.root, @(y) x.add(ipermute(y.grad,dimorder),ipermute(y.grad_c,dimorder)));
            if x.real_valued
                y.real_valued=true;
            end
            if ~x.reuse
                x.value=[];%save memory
            end
        end

        function y = fftshift(x, dim)
            switch nargin
                case 1
                    y = ADNode(fftshift(x.value), x.root, @(y) x.add(ifftshift(y.grad),ifftshift(y.grad_c)));
                otherwise
                    y = ADNode(fftshift(x.value,dim), x.root, @(y) x.add(ifftshift(y.grad,dim),ifftshift(y.grad_c,dim)));
            end
            if x.real_valued
                y.real_valued=true;
            end
            if ~x.reuse
                x.value=[];%save memory
            end
        end
        function y = ifftshift(x, dim)
            switch nargin
                case 1
                    y = ADNode(ifftshift(x.value), x.root, @(y) x.add(fftshift(y.grad),fftshift(y.grad_c)));
                otherwise
                    y = ADNode(ifftshift(x.value,dim), x.root, @(y) x.add(fftshift(y.grad,dim),fftshift(y.grad_c,dim)));
            end
            if x.real_valued
                y.real_valued=true;
            end
            if ~x.reuse
                x.value=[];%save memory
            end
        end
        
        function y = fft2(x)
            y = ADNode(fft2(x.value), x.root, @(y) x.add((fft2(y.grad)),(fft2(y.grad_c))));
            if ~x.reuse
                x.value=[];%save memory
            end
        end
        function y = s_fft2(x)
            y = ADNode(s_fft2(x.value), x.root, @(y) x.add((s_fft2(y.grad)),(s_fft2(y.grad_c))));
            if ~x.reuse
                x.value=[];%save memory
            end
        end
        function y = sn_fft2(x)
            y = ADNode(sn_fft2(x.value), x.root, @(y) x.add((sn_fft2(y.grad)),(sn_fft2(y.grad_c))));
            if ~x.reuse
                x.value=[];%save memory
            end
        end
        function y = ifft2(x)
            y = ADNode(ifft2(x.value), x.root, @(y) x.add((ifft2(y.grad)),(ifft2(y.grad_c))));
            if ~x.reuse
                x.value=[];%save memory
            end
        end
        function y = s_ifft2(x)
            y = ADNode(s_ifft2(x.value), x.root, @(y) x.add((s_ifft2(y.grad)),(s_ifft2(y.grad_c))));
            if ~x.reuse
                x.value=[];%save memory
            end
        end
        function y = sn_ifft2(x)
            y = ADNode(sn_ifft2(x.value), x.root, @(y) x.add((sn_ifft2(y.grad)),(sn_ifft2(y.grad_c))));
            if ~x.reuse
                x.value=[];%save memory
            end
        end
        
        function y = fft3(x)
            y = ADNode(fft3(x.value), x.root, @(y) x.add((fft3(y.grad)),(fft3(y.grad_c))));
            if ~x.reuse
                x.value=[];%save memory
            end
        end
        function y = s_fft3(x)
            y = ADNode(s_fft3(x.value), x.root, @(y) x.add((s_fft3(y.grad)),(s_fft3(y.grad_c))));
            if ~x.reuse
                x.value=[];%save memory
            end
        end
        function y = sn_fft3(x)
            y = ADNode(sn_fft3(x.value), x.root, @(y) x.add((sn_fft3(y.grad)),(sn_fft3(y.grad_c))));
            if ~x.reuse
                x.value=[];%save memory
            end
        end
        function y = ifft3(x)
            y = ADNode(ifft3(x.value), x.root, @(y) x.add((ifft3(y.grad)),(ifft3(y.grad_c))));
            if ~x.reuse
                x.value=[];%save memory
            end
        end
        function y = s_ifft3(x)
            y = ADNode(s_ifft3(x.value), x.root, @(y) x.add((s_ifft3(y.grad)),(s_ifft3(y.grad_c))));
            if ~x.reuse
                x.value=[];%save memory
            end
        end
        function y = sn_ifft3(x)
            y = ADNode(sn_ifft3(x.value), x.root, @(y) x.add((sn_ifft3(y.grad)),(sn_ifft3(y.grad_c))));
            if ~x.reuse
                x.value=[];%save memory
            end
        end
        
        function y = fftn(x)
            y = ADNode(fftn(x.value), x.root, @(y) x.add((fftn(y.grad)),(fftn(y.grad_c))));
            if ~x.reuse
                x.value=[];%save memory
            end
        end
        function y = s_fftn(x)
            y = ADNode(s_fftn(x.value), x.root, @(y) x.add((s_fftn(y.grad)),(s_fftn(y.grad_c))));
            if ~x.reuse
                x.value=[];%save memory
            end
        end
        function y = sn_fftn(x)
            y = ADNode(sn_fftn(x.value), x.root, @(y) x.add((sn_fftn(y.grad)),(sn_fftn(y.grad_c))));
            if ~x.reuse
                x.value=[];%save memory
            end
        end
        function y = ifftn(x)
            y = ADNode(ifftn(x.value), x.root, @(y) x.add((ifftn(y.grad)),(ifftn(y.grad_c))));
            if ~x.reuse
                x.value=[];%save memory
            end
        end
        function y = s_ifftn(x)
            y = ADNode(s_ifftn(x.value), x.root, @(y) x.add((s_ifftn(y.grad)),(s_ifftn(y.grad_c))));
            if ~x.reuse
                x.value=[];%save memory
            end
        end
        function y = sn_ifftn(x)
            y = ADNode(sn_ifftn(x.value), x.root, @(y) x.add((sn_ifftn(y.grad)),(sn_ifftn(y.grad_c))));
            if ~x.reuse
                x.value=[];%save memory
            end
        end
        
        function y = fft(x,n,dim)
            y = ADNode(fft(x.value,n,dim), x.root, @(y) x.add((fft(y.grad,n,dim)),(fft(y.grad_c,n,dim))));
            if ~x.reuse
                x.value=[];%save memory
            end
        end
        function y = s_fft(x,dim)
            y = ADNode(s_fft(x.value,dim), x.root, @(y) x.add((s_fft(y.grad,dim)),(s_fft(y.grad_c,dim))));
            if ~x.reuse
                x.value=[];%save memory
            end
        end
        function y = sn_fft(x,dim)
            y = ADNode(sn_fft(x.value,dim), x.root, @(y) x.add((sn_fft(y.grad,dim)),(sn_fft(y.grad_c,dim))));
            if ~x.reuse
                x.value=[];%save memory
            end
        end
        function y = ifft(x,n,dim)
            y = ADNode(ifft(x.value,n,dim), x.root, @(y) x.add((ifft(y.grad,n,dim)),(ifft(y.grad_c,n,dim))));
            if ~x.reuse
                x.value=[];%save memory
            end
        end
        function y = s_ifft(x,dim)
            y = ADNode(s_ifft(x.value,dim), x.root, @(y) x.add((s_ifft(y.grad,dim)),(s_ifft(y.grad_c,dim))));
            if ~x.reuse
                x.value=[];%save memory
            end
        end
        function y = sn_ifft(x,dim)
            y = ADNode(sn_ifft(x.value,dim), x.root, @(y) x.add((sn_ifft(y.grad,dim)),(sn_ifft(y.grad_c,dim))));
            if ~x.reuse
                x.value=[];%save memory
            end
        end
        
        function y = mean(x, dim, flag)
            if nargin==1
                dim=1;
            elseif strcmp(dim,'all' )
                dim=1:length(size(x.value));
            end
            sz_var=size(x.value);
            avg_size=prod(sz_var(dim),'all');
            
            switch nargin
                case 3
                    y = ADNode(mean(x.value, dim, flag), x.root, @(y) x.add(rep_to_good_size(y.grad./avg_size,sz_var),rep_to_good_size(y.grad_c./avg_size,sz_var)));
                    if ~x.reuse
                        x.value=[];
                    end
                case 2
                    y = ADNode(mean(x.value, dim), x.root, @(y) x.add(rep_to_good_size(y.grad./avg_size,sz_var),rep_to_good_size(y.grad_c./avg_size,sz_var)));
                    if ~x.reuse
                        x.value=[];
                    end
                otherwise
                    y = ADNode(mean(x.value), x.root, @(y) x.add(rep_to_good_size(y.grad./avg_size,sz_var),rep_to_good_size(y.grad_c./avg_size,sz_var)));
                    if ~x.reuse
                        x.value=[];
                    end
            end
            if x.real_valued
                y.real_valued=true;
            end
        end

        function y = sum(x, dim, flag)
            sz_var=size(x.value);
            %rep_to_good_size(y.grad,sz_var),rep_to_good_size(y.grad_c,sz_var)
            switch nargin
                case 3
                    y = ADNode(sum(x.value, dim, flag), x.root, @(y) x.add(rep_to_good_size(y.grad,sz_var),rep_to_good_size(y.grad_c,sz_var)));
                    if ~x.reuse
                        x.value=[];
                    end
                case 2
                    y = ADNode(sum(x.value, dim), x.root, @(y) x.add(rep_to_good_size(y.grad,sz_var),rep_to_good_size(y.grad_c,sz_var)));
                    if ~x.reuse
                        x.value=[];
                    end
                otherwise
                    y = ADNode(sum(x.value), x.root, @(y) x.add(rep_to_good_size(y.grad,sz_var),rep_to_good_size(y.grad_c,sz_var)));
                    if ~x.reuse
                        x.value=[];
                    end
            end
            if x.real_valued
                y.real_valued=true;
            end
        end
        function y = abs(x)
            %sqrt(a*conj(a))
            %debug_sz(matt) // debug_matt(matt)
            y = ADNode(abs(x.value), x.root,@(y) x.add(...
                bsxfun2(@plus,bsxfun2(@times, (y.grad)  , 0.5.*sign(conj(x.value))),bsxfun2(@times, conj(y.grad_c),0.5.*sign(conj(x.value)))),...
                bsxfun2(@plus,bsxfun2(@times,  (y.grad_c), 0.5.*sign(conj(x.value))),bsxfun2(@times, conj(y.grad)  ,0.5.*sign(conj(x.value)))) ));
            y.real_valued=true;
        end
        function y = abs2(x)
            %(a*conj(a))
            %debug_sz(matt) // debug_matt(matt)
            y = ADNode(abs2(x.value), x.root,@(y) x.add(...
                bsxfun2(@plus,bsxfun2(@times, (y.grad)  , (conj(x.value))),bsxfun2(@times, conj(y.grad_c),(conj(x.value)))),...
                bsxfun2(@plus,bsxfun2(@times,  (y.grad_c), (conj(x.value))),bsxfun2(@times, conj(y.grad)  ,(conj(x.value)))) ));
            y.real_valued=true;
        end

        function y = real(x)
            y = ADNode(real(x.value), x.root,@(y) x.add(...
                set_real_flag(0.5.*bsxfun2(@plus,y.grad  ,conj(y.grad_c)),x.root.real_optimization),...
                set_real_flag(0.5.*bsxfun2(@plus,y.grad_c,conj(y.grad)  ),x.root.real_optimization) ));
            y.real_valued=true;
            if ~x.reuse
                x.value=[];%save memory
            end
        end
        function y = imag(x)
            y = ADNode(imag(x.value), x.root,@(y) x.add(...
                -0.5i.*bsxfun2(@plus,y.grad  ,conj(y.grad_c)),...
                -0.5i.*bsxfun2(@plus,y.grad_c,conj(y.grad)  )));
            if ~x.reuse
                x.value=[];%save memory
            end
        end

        function y = conj(x)
            %sqrt(a*conj(a))
            %debug_sz(matt) // debug_matt(matt)
            y = ADNode(conj(x.value), x.root,@(y) x.add(...
                conj(y.grad_c),...
                conj(y.grad)));
            if x.real_valued
                y.real_valued=true;
            end
            if ~x.reuse
                x.value=[];%save memory
            end
        end

        function y = reusable(x)
            x.root.tape=x.root.tape(1:end-1);
            y = ADNode(x.value, x.root,x.func);
            y.reuse=true;
            if x.real_valued
                y.real_valued=true;
            end
            if ~x.reuse
                x.value=[];%save memory
            end
        end

        function y = power(x1, x2)
            if isa(x1, 'ADNode')
                if isa(x2, 'ADNode')
                    y = ADNode(x1.value .^ x2.value, x1.root, @(y) y.power_backprop(x1, x2));
                else
                    switch x2
                        case 1
                            y = ADNode(x1.value .^ x2, x1.root, @(y) x1.add(y.grad,y.grad_c));
                        case 2
                            y = ADNode(x1.value .^ x2, x1.root, @(y) x1.add_times_holo(y,( x1.value * 2)));
                        otherwise
                            y = ADNode(x1.value .^ x2, x1.root, @(y) x1.add_times_holo(y,( x1.value .^ (x2-1) .* x2 )));
                    end
                    if x1.real_valued && isreal(x2) && floor (x2)==x2 %check if an integer 
                        y.real_valued=true;
                    end
                end
            else
                t = x1 .^ x2.value;
                y = ADNode(t, x2.root, @(y) x2.add(bsxfun2(@times, y.grad, t .* log(x1))));
            end
        end
        function y = sqrt(x)
            y = ADNode(sqrt(x.value), x.root, @(y) x.add_times_holo(y, 1./(2*sqrt(x.value))));
            if x.real_valued
                y.real_valued=true;
            end
        end
        function y = tanh(x)
            y = ADNode(tanh(x.value), x.root, @(y) x.add_times_holo(y, sech(x.value) .^ 2));
        end
        function y = acos(x)
            y = ADNode(acos(x.value), x.root, @(y) x.add_times_holo(y, -sqrt(1-x.value.^2)));
        end
        function y = asin(x)
            y = ADNode(asin(x.value), x.root, @(y) x.add_times_holo(y,  sqrt(1-x.value.^2)));
        end
        function y = atan(x)
            y = ADNode(atan(x.value), x.root, @(y) x.add_times_holo(y,(1+x.value.^2)));
        end
        function y = atan2(x1,x2)
            y = ADNode(atan2(x1.value,x2.value), x1.root, @(y) y.atan2_backprop(x1, x2));
        end
        function y = cos(x)
            y = ADNode(cos(x.value), x.root, @(y) x.add_times_holo(y,-sin(x.value)));
            if x.real_valued
                y.real_valued=true;
            end
        end
        function y = exp(x)
            y = ADNode(exp(x.value), x.root, @(y) x.add_times_holo(y, exp(x.value)));
            if x.real_valued
                y.real_valued=true;
            end
        end
        function y = log(x)
            y = ADNode(log(x.value), x.root, @(y) x.add_times_holo(y,1./ x.value));
        end
        function y = sin(x)
            y = ADNode(sin(x.value), x.root, @(y) x.add_times_holo(y, cos(x.value)));
            if x.real_valued
                y.real_valued=true;
            end
        end
        function y = tan(x)
            y = ADNode(tan(x.value), x.root, @(y) x.add_times_holo(y, sec(x.value) .^ 2));
            if x.real_valued
                y.real_valued=true;
            end
        end

        function y = uminus(x)
            y = minus(0,x);
        end

        function y = uplus(x)
            y = plus(0,x);
        end

        function [varargout] = subsref(x, s)
            switch s(1).type
                case '()'
                    y= ADNode(x.value(s.subs{:}), x.root, @(y) x.subs_add(s.subs, y));
                    if x.real_valued
                        y.real_valued=true;
                    end
                    varargout{1} =y;
                otherwise
                    [varargout{1:nargout}] = builtin('subsref', x, s);
            end
        end

        function y = subsasgn(x, s, varargin)
            switch s(1).type
                case '()'
                    if isa(varargin{1}, 'ADNode')
                        x.value(s.subs{:}) = varargin{1}.value;
                        t = ADNode(x.value(s.subs{:}), x.root, @(y) varargin{1}.subs_move(s.subs, x));
                        y = x;
                        if x.real_valued
                            y.real_valued=true;
                        end
                    else
                        x.value(s.subs{:}) = varargin{1};
                        t = ADNode(x.value(s.subs{:}), x.root, @(y) x.subs_clear(s.subs));
                        y = x;
                        if x.real_valued
                            y.real_valued=true;
                        end
                    end
                    %{
                case '.'
                    x.(s(1).subs)=varargin{1};
                    y=x;
                    %}
                otherwise
                    y = builtin('subsagn', x, s, varargin);
                    if x.real_valued
                        y.real_valued=true;
                    end
            end
        end

        function y = plus(x1, x2)
            if isa(x1, 'ADNode')
                if isa(x2, 'ADNode')
                    sz1=size(x1.value);
                    sz2=size(x2.value);
                    y = ADNode(bsxfun2(@plus, x1.value, x2.value), x1.root, @(y) y.plus_backprop(x1, x2,sz1,sz2));
                    if x2.real_valued && x1.real_valued
                        y.real_valued=true;
                    end
                    if ~x1.reuse
                        x1.value=[];
                    end
                    if ~x2.reuse
                        x2.value=[];
                    end
                else
                    sz1=size(x1.value);
                    y = ADNode(bsxfun2(@plus, x1.value, x2), x1.root, @(y) x1.add_compress(sz1,y.grad,y.grad_c));
                    if x1.real_valued && isreal(x2)
                        y.real_valued=true;
                    end
                    if ~x1.reuse
                        x1.value=[];
                    end
                end
            else
                sz2=size(x2.value);
                y = ADNode(bsxfun2(@plus, x1, x2.value), x2.root, @(y) x2.add_compress(sz2,y.grad,y.grad_c));
                if x2.real_valued && isreal(x1)
                    y.real_valued=true;
                end
                if ~x2.reuse
                    x2.value=[];
                end
            end
        end
        function y = minus(x1, x2)
            if isa(x1, 'ADNode')
                if isa(x2, 'ADNode')
                    sz1=size(x1.value);
                    sz2=size(x2.value);
                    y = ADNode(bsxfun2(@minus, x1.value, x2.value), x1.root, @(y) y.minus_backprop(x1, x2,sz1,sz2));
                    if x2.real_valued && x1.real_valued
                        y.real_valued=true;
                    end
                    if ~x1.reuse
                        x1.value=[];
                    end
                    if ~x2.reuse
                        x2.value=[];
                    end
                else
                    sz1=size(x1.value);
                    y = ADNode(bsxfun2(@minus, x1.value, x2), x1.root, @(y) x1.add_compress(sz1,y.grad,y.grad_c));
                    if x1.real_valued && isreal(x2)
                        y.real_valued=true;
                    end
                    if ~x1.reuse
                        x1.value=[];
                    end
                end
            else
                sz2=size(x2.value);
                y = ADNode(bsxfun2(@minus, x1, x2.value), x2.root, @(y) x2.add_compress(sz2,-y.grad,-y.grad_c));
                if x2.real_valued && isreal(x1)
                    y.real_valued=true;
                end
                if ~x2.reuse
                    x2.value=[];
                end
            end
        end
        function y = mtimes(x1, x2)
            if isa(x1, 'ADNode')
                if isa(x2, 'ADNode')
                    y = ADNode(x1.value * x2.value, x1.root, @(y) y.mtimes_backprop(x1, x2));
                    if x2.real_valued && x1.real_valued
                        y.real_valued=true;
                    end
                else
                    y = ADNode(x1.value * x2, x1.root, @(y) x1.add( y.grad * (x2.'), y.grad * (x2.')));
                    if x1.real_valued && isreal(x2)
                        y.real_valued=true;
                    end
                    if ~x1.reuse
                        x1.value=[];
                    end
                end
            else
                y = ADNode(x1 * x2.value, x2.root, @(y) x2.add( (x1.') * y.grad,(x1.') * y.grad));
                if x2.real_valued && isreal(x1)
                    y.real_valued=true;
                end
                if ~x2.reuse
                    x2.value=[];
                end
            end
        end

        function y = times(x1, x2)
            if isa(x1, 'ADNode')
                if isa(x2, 'ADNode')
                    sz1=size(x1.value);
                    sz2=size(x2.value);
                    y = ADNode(bsxfun2(@times, x1.value, x2.value), x1.root, @(y) y.times_backprop(x1, x2,sz1,sz2));
                    if x2.real_valued && x1.real_valued
                        y.real_valued=true;
                    end
                else
                    sz1=size(x1.value);
                    y = ADNode(bsxfun2(@times, x1.value, x2), x1.root, @(y) x1.add_compress(sz1,bsxfun2(@times, y.grad,(x2)),bsxfun2(@times, y.grad_c,(x2))));
                    if x1.real_valued && isreal(x2)
                        y.real_valued=true;
                    end
                    if ~x1.reuse
                        x1.value=[];
                    end
                end
            else
                sz2=size(x2.value);
                y = ADNode(bsxfun2(@times, x1, x2.value), x2.root, @(y) x2.add_compress(sz2,bsxfun2(@times, y.grad, (x1)),bsxfun2(@times, y.grad_c, (x1))));
                if x2.real_valued && isreal(x1)
                    y.real_valued=true;
                end
                if ~x2.reuse
                    x2.value=[];
                end
            end
        end

        function y = rdivide(x1, x2)
            if isa(x1, 'ADNode')
                if isa(x2, 'ADNode')
                    sz1=size(x1.value);
                    sz2=size(x2.value);
                    y = ADNode(bsxfun2(@rdivide, x1.value, x2.value), x1.root, @(y) y.rdivide_backprop(x1, x2,sz1,sz2));
                    if x1.real_valued && x2.real_valued
                        y.real_valued=true;
                    end
                else
                    sz1=size(x1.value);
                    y = ADNode(bsxfun2(@rdivide, x1.value, x2), x1.root, ...
                        @(y) x1.add_compress(sz1,bsxfun2(@rdivide, y.grad, x2),bsxfun2(@rdivide, y.grad_c, x2)));
                    if ~isequal(size(x1.value),size(x2))
                        error('size changed during div implement');
                    end
                    if x1.real_valued && isreal(x2)
                        y.real_valued=true;
                    end
                    if ~x1.reuse
                        x1.value=[];
                    end
                end
            else
                sz2=size(x2.value);
                %y = ADNode(bsxfun2(@rdivide, x1, x2.value), x2.root, ...
                %    @(y) x2.add_compress(sz2,- y.grad .* bsxfun2(@rdivide, x1, x2.value .^ 2),- y.grad_c .* bsxfun2(@rdivide, x1, x2.value .^ 2)));
                y = ADNode(bsxfun2(@rdivide, x1, x2.value), x2.root, ...
                    @(y) x2.add_compress(sz2,- bsxfun2(@times,y.grad, bsxfun2(@rdivide, x1, x2.value .^ 2)),- bsxfun2(@times,y.grad_c, bsxfun2(@rdivide, x1, x2.value .^ 2))));
                if x2.real_valued && isreal(x1)
                    y.real_valued=true;
                end
            end
        end

        function y = mrdivide(x1, x2)

            error('not implemeted mrdivide')
            if isa(x1, 'ADNode')
                if isa(x2, 'ADNode')
                    y = ADNode(x1.value / x2.value, x1.root, @(y) y.mrdivide_backprop(x1, x2));
                else
                    y = ADNode(x1.value / x2, x1.root, @(y) x1.add(y.grad / x2));
                end
            else
                y = ADNode(x1 / x2.value, x2.root, @(y) x2.add(- y.grad .* x1 / x2.value .^ 2));
            end
        end

        function y = mldivide(x1, x2)

            error('not implemeted mldivide')
            if isa(x1, 'ADNode')
                if isa(x2, 'ADNode')
                    y = ADNode(x1.value\x2.value, x1.root, @(y) y.mldivide_backprop(x1, x2));
                else
                    y = ADNode(x1.value\x2, x1.root, @(y) x1.add(y.grad .* ((x1.value^2)\eye(size(x1.value)) * x2) ) );
                end
            else
                y = ADNode(x1\x2.value, x2.root, @(y) x2.add(y.grad .* (x1\y.grad) ) );
            end
        end

        function y = mpower(x1, x2)

            error('not implemeted mpower')
            if isa(x1, 'ADNode')
                if isa(x2, 'ADNode')
                    y = ADNode(x1.value ^ x2.value, x1.root, @(y) y.mpower_backprop(x1, x2));
                else
                    switch x2
                        case 1
                            y = ADNode(x1.value ^ x2, x1.root, @(y) x1.add(y.grad));
                            if ~x1.reuse
                                x1.value=[];
                            end
                        case 2
                            y = ADNode(x1.value ^ x2, x1.root, @(y) x1.add(y.grad * x1.value * 2));
                        otherwise
                            y = ADNode(x1.value ^ x2, x1.root, @(y) x1.add(y.grad * x1.value ^ (x2-1) * x2));
                    end
                end
            else
                t = x1 ^ x2.value;
                y = ADNode(t, x2.root, @(y) x2.add(y.grad * t * log(x1)));
            end
        end

        function y = length(adn)
            y = length(adn.value);
        end

        function y = size(adn, dim)
            if nargin < 2;
                y = size(adn.value);
            else
                y = size(adn.value, dim);
            end
        end

        function y = bsxfun(op, x1, x2)
            switch func2str(op)
                case 'minus'
                    y = minus(x1, x2);
                case 'plus'
                    y = plus(x1, x2);
                case 'times'
                    y = times(x1, x2);
                case 'rdivide'
                    y = rdivide(x1, x2);
                otherwise
                    assert(false, 'not implemented');
            end
        end

        function y = min(x1, x2)

            error('not implemeted min')
            if nargin < 2
                [m, k] = min(x1.value);
                y = ADNode(m, x1.root, @(y) x1.subs_add({k}, y));
            else
                if isa(x1, 'ADNode')
                    if isa(x2, 'ADNode')
                        m = min(x1.value, x2.value);
                        y = ADNode(m, x1.root, @(y) y.minmax_backprop(x1, x2));
                    else
                        m = min(x1.value, x2);
                        y = ADNode(m, x1.root, @(y) x1.subs_match({find(m == x1.value)}, y));
                    end
                else
                    m = min(x1, x2.value);
                    y = ADNode(m, x2.root, @(y) x2.subs_match({find(m == x2.value)}, y));
                end
            end
        end

        function y = max(x1, x2)

            error('not implemeted max')
            if nargin < 2
                [m, k] = max(x1.value);
                y = ADNode(m, x1.root, @(y) x1.subs_add({k}, y));
            else
                if isa(x1, 'ADNode')
                    if isa(x2, 'ADNode')
                        m = max(x1.value, x2.value);
                        y = ADNode(m, x1.root, @(y) y.minmax_backprop(x1, x2));
                    else
                        m = max(x1.value, x2);
                        y = ADNode(m, x1.root, @(y) x1.subs_match({find(m == x1.value)}, y));
                    end
                else
                    m = max(x1, x2.value);
                    y = ADNode(m, x2.root, @(y) x2.subs_match({find(m == x2.value)}, y));
                end
            end
        end

        function y = norm(x, d)
            if (nargin==1) d = 2; end
            y = sum(abs(x) .^ d) .^ (1/d);
        end

        function y = end(adn, dim, n)
            if n == 1
                y = length(adn.value);
            else
                y = size(adn.value, dim);
            end
        end

        function y = transpose(x)
            y = ADNode(x.value.', x.root, @(y) x.add( y.grad.' ));
            if ~x.reuse
                x.value=[];
            end
        end

        function y = ctranspose(x)
            y = conj(transpose(x));
        end

        % eq
        % ge
        % gt
        % le
        % lt
        % ne
        % sort
        % vertcat
        % horzcat

    end

    methods (Access = private)
        function add_times_holo(x,y, direct_grad)
            x.add(bsxfun2(@times, y.grad, direct_grad),bsxfun2(@times, y.grad_c, direct_grad));
        end
        function add_compress(x,sz, grad,grad_c)
            x.add(compress_sz(grad,sz),compress_sz(grad_c,sz));
        end
        function add(x, grad,grad_c)
            %% accumulate the gradient, take sum of dimensions if needed

            temp=grad;
            if ~isempty(temp)
                if isempty(x.grad)
                    x.grad=gradient_typing(size(x.value));
                end
                for ii=1:length(size(x.value))
                    if size(x.value, ii)==1
                        temp=sum(temp,ii);
                    end
                end
                if ~isempty(x.grad)
                    x.grad = x.grad+temp;
                else
                    x.grad = temp;
                end
            end

            temp=grad_c;
            if ~isempty(temp)
                if isempty(x.grad)
                    x.grad_c=gradient_typing(size(x.value));
                end
                for ii=1:length(size(x.value))
                    if size(x.value, ii)==1
                        temp=sum(temp,ii);
                    end
                end
                if ~isempty(x.grad_c)
                    x.grad_c = x.grad_c+temp;
                else
                    x.grad_c = temp;
                end
            end

        end

        function subs_add(x, subs, y)
            %% accumulate the gradient with subscripts
            grad = y.grad;
            grad_c = y.grad_c;


            if ~isempty(grad)
                if isempty(x.grad)
                    x.grad = gradient_typing(size(x.value));
                end
                old = x.grad(subs{:});
                for ii=1:max(length(size(old)),length(size(grad)))
                    if size(old, ii)==1
                        grad=sum(grad,ii);
                    end
                end
                x.grad(subs{:}) = old+grad;
            end

            if ~isempty(grad_c)
                if isempty(x.grad_c)
                    x.grad_c = gradient_typing(size(x.value));
                end
                old = x.grad_c(subs{:});
                for ii=1:max(length(size(old)),length(size(grad_c)))
                    if size(old, ii)==1
                        grad_c=sum(grad_c,ii);
                    end
                end
                x.grad_c(subs{:}) = old+grad_c;
            end
        end

        function subs_match(x, subs, y)
            error('not implemented')
            %% accumulate the gradient with subscripts
            if isempty(x.grad)
                x.grad = gradient_typing(size(x.value));
            end
            if size(x.grad) == [1, 1]
                x.grad = x.grad + sum(y.grad(subs{:}));
            else
                x.grad(subs{:}) = x.grad(subs{:}) + y.grad(subs{:});
            end
        end

        function subs_clear(x, subs)
            %% clear the gradient with subscripts
            if isempty(x.grad)
                x.grad = gradient_typing(size(x.value));
            end
            if isempty(x.grad_c)
                x.grad_c = gradient_typing(size(x.value));
            end
            x.grad(subs{:}) = 0;
            x.grad_c(subs{:}) = 0;
        end

        function subs_move(x, subs, y)
            %% accumulate the gradient with subscripts
            %if size(y.grad) == [1,1]; y.grad = repmat(y.grad, size(y.value)); end
            %error('implement both grad and grad_c')
            grad = y.grad(subs{:});
            grad_c = y.grad_c(subs{:});
            y.grad(subs{:}) = 0;
            y.grad_c(subs{:}) = 0;
            if isempty(x.grad) && ~isempty(grad)
                x.grad = gradient_typing(size(x.value));
            end
            if isempty(x.grad_c) && ~isempty(grad_c)
                x.grad_c = gradient_typing(size(x.value));
            end
            for mm=1:length(size(x.grad))
               if size(x.grad,mm)==1
                   grad=sum(grad,mm);
               end
            end
            for mm=1:length(size(x.grad_c))
               if size(x.grad_c,mm)==1
                   grad_c=sum(grad_c,mm);
               end
            end
            x.grad = x.grad + grad;
            x.grad_c = x.grad_c + grad_c;
            
        end

        function plus_backprop(y, x1, x2,sz1,sz2)
            x1.add_compress(sz1,y.grad,y.grad_c);
            x2.add_compress(sz2,y.grad,y.grad_c);
        end

        function minus_backprop(y, x1, x2,sz1,sz2)
            x1.add_compress(sz1,y.grad,y.grad_c);
            x2.add_compress(sz2,-y.grad,-y.grad_c);
        end

        function atan2_backprop(y,x1,x2)
            x1.add_times_holo(y, x2.value ./ (x2.value.^2 + x1.value.^2));
            x2.add_times_holo(y, -x1.value ./ (x2.value.^2 + x1.value.^2));
        end

        function mtimes_backprop(y, x1, x2)
            x1.add( y.grad * (x2.value.'), y.grad_c * (x2.value.'));
            x2.add( (x1.value.') * y.grad, (x1.value.') * y.grad_c);
        end

        function times_backprop(y, x1, x2,sz1,sz2)
            x1.add_compress(sz1,bsxfun2(@times, y.grad, x2.value),bsxfun2(@times, y.grad_c, x2.value));
            x2.add_compress(sz2,bsxfun2(@times, y.grad, x1.value),bsxfun2(@times, y.grad_c, x1.value));
        end

        function rdivide_backprop(y, x1, x2,sz1,sz2)
            x1.add_compress(sz1,bsxfun2(@rdivide, y.grad, x2.value),bsxfun2(@rdivide, y.grad_c, x2.value));
            x2.add_compress(sz2,-y.grad .* bsxfun2(@rdivide, x1.value, x2.value .^ 2),-y.grad_c .* bsxfun2(@rdivide, x1.value, x2.value .^ 2));
        end

        function mrdivide_backprop(y, x1, x2)
            x1.add(y.grad / x2.value);
            x2.add(-y.grad .* x1.value / x2.value .^ 2);
        end

        function mldivide_backprop(y, x1, x2)
            %x1.add(y.grad .* ((x1.value^2)\eye(size(x1.value)) * x2.value) );
            %x2.add(y.grad .* (x1.value\y.grad) );
            x1.add(-(x1.value')\(y.grad*y.value') );
            x2.add(x1.value'\y.grad);
            %disp('asdf');
        end

        function mpower_backprop(y, x1, x2)
            x1.add(y.grad * x1.value ^ (x2.value-1) * x2.value);
            x2.add(y.grad * y.value * log(x1.value));
        end

        function power_backprop(y, x1, x2)
            x1.add(y.grad .* x1.value .^ (x2.value-1) .* x2.value);
            x2.add(y.grad .* y.value .* log(x1.value));
        end

        function minmax_backprop(y, x1, x2)
            x1.subs_match({find(y.value == x1.value)}, y);
            x2.subs_match({find(y.value == x2.value)}, y);
        end

    end

end
function y = bsxfun2(op, x1, x2)
opp=func2str(op);
y=[];
if (strcmp(opp,'times'))&&(isempty(x1)||isempty(x2))
    return;
end
if (strcmp(opp,'rdivide'))&&(isempty(x1)||isempty(x2))
    if isempty(x1)
        return;
    end
    error('right term must not be empty');
    return
end
if strcmp(opp,'plus')&&(isempty(x1)||isempty(x2))
    if isempty(x1)
        y=x2;
        return
    end
    if isempty(x2)
        y=x1;
        return
    end
    return
end
if strcmp(opp,'minus')&&(isempty(x1)||isempty(x2))
    if isempty(x1)
        y=-x2;
        return
    end
    if isempty(x2)
        y=x1;
        return
    end
    return
end

y = bsxfun(op, x1, x2);
end

function x=gradient_typing(sz)
x=zeros(sz,'single','gpuArray');
end

function matt=debug_sz(matt)
display(['size : ' num2str(size(matt))]);
end
function matt=debug_matt(matt)
matt
%figure; imagesc(angle(matt));
end

function x=compress_sz(x, sz)
if ~isequal(sz,size(x))
    %display('compress')
    for ii=1:max(length(size(x)),length(sz))
        if ii>length(sz) || sz(ii)==1
            x=sum(x,ii);
        end
    end
end
end
function x=set_real_flag(x,flag)
if flag
    x=real(x);
end
end
function array=rep_to_good_size(array,size_goal)
if isempty(array)
    return
end
repetitions=1+0.*size_goal(:);
for ii=1:length(size_goal(:))
    if size_goal(ii)~=size(array, ii)
        if size(array, ii)~=1
            error(['unexpected expansion : sz ' num2str(size(array, ii))]);
        end
        repetitions(ii)=size_goal(ii);
    end
end
array = repmat(array,repetitions(:)');
end
function x=reshape_if_not_empty(x,sz)
if ~isempty(x)
    x=reshape(x,sz);
end
end
