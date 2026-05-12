function name = pretty_covariance_method_name(method)
%% Map internal covariance-refit ids to concise display names.
%
% Usage:
%   name = pretty_covariance_method_name("ec2");
%
% Input:
%   @method: Internal method id.
%
% Output:
%   @name: Short human-readable label.

    switch string(method)
        case "ec2"
            name = 'EC2';
        case "xue_ma_zou"
            name = 'Xue-Ma-Zou';
        case "adaptive_threshold"
            name = 'Adaptive Thresh';
        case "adaptive_threshold_fspd"
            name = 'Adaptive+FSPD';
        case "convex_banding"
            name = 'Convex Banding';
        case {"sample", "sample_covariance"}
            name = 'Sample';
        otherwise
            name = char(method);
    end
end
