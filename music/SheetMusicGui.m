classdef SheetMusicGui < handle

properties (Access = private)
f  % Figure handle
h  % Axes handle

% Coordinates of the box
topLineY    = 0.8;
bottomLineY = 0.2;

leftLineX   = 0.1;
maxX        = 3;
end

methods

function obj = SheetMusicGui(varargin)

% Initialize figure and axis
obj.f = figure; clf, hold on
set(obj.f,...
    'position', [400 400 900 300]);

obj.h = gca;
set(obj.h, ...
    'box', 'on',...
    'xtick',[], 'xticklabel',[],...
    'ytick',[], 'yticklabel',[]);

% Draw the grande staff
obj.drawGrandeStaff(obj.h);

% Make sure all measurements work out
axis([0 3 0 1]);
end

end

methods (Access = private)

%% The Basics

function [h] = drawGrandeStaff(obj, h)

% First bar line
line([obj.leftLineX obj.leftLineX], [obj.bottomLineY obj.topLineY], 'color', 'k');

% Brace
img = imread('brace.png');

X = size(img,2);    xExtent = 0.068;
Y = size(img,1);    yExtent = xExtent/X*Y;

C = imagesc([0.01 0.01+xExtent],[0.8 0.8-yExtent],img, 'parent',obj.h);
uistack(C, 'bottom'); % At the bottom prevents transparency issues


% Treble staff
line(...
     repmat([obj.leftLineX; obj.maxX],1,5), ...
     repmat(linspace(obj.topLineY, obj.topLineY-0.15, 5), 2,1), 'color','k')
% Treble clef
obj.drawTrebleClef(obj.leftLineX+0.04, obj.topLineY+0.05);


% Bass staff
line(...
     repmat([obj.leftLineX; obj.maxX],1,5),...
     repmat(linspace(obj.bottomLineY, obj.bottomLineY+0.15, 5), 2,1), 'color','k')

% Bass clef
obj.drawBassClef(obj.leftLineX+0.04, obj.bottomLineY+0.155);

end

% Draw a G clef at location X,Y
function C = drawTrebleClef(obj, x, y)
persistent img
if isempty(img)
img = imread('GClef.png'); end

% Scale image
X = size(img,2);    xExtent = 0.1;
Y = size(img,1);    yExtent = xExtent/X*Y;

% Plage image
C = imagesc([x x+xExtent],[y y-yExtent],img, 'parent',obj.h);
uistack(C, 'bottom'); % At the bottom prevents transparency issues

end

% Draw an F clef at location X,Y
function C = drawBassClef(obj, x, y)
persistent img
if isempty(img)
img = imread('FClef.png'); end

% Scale image
X = size(img,2);    xExtent = 0.12;
Y = size(img,1);    yExtent = xExtent/X*Y;

% Plage image
C = imagesc([x x+xExtent],[y y-yExtent],img, 'parent',obj.h);
uistack(C, 'bottom'); % At the bottom prevents transparency issues

end

function T = drawTime(obj, varargin)

% TODO

% NOTE: you had best use some pre-defined times, like common time,
% 2/2, 3/4, 6/8, etc. It looks much nicer. Only use text() when a
% non-common time is encountered.
end

function K = drawKey(obj, varargin)
% TODO
% NOTE: use sharp/flat below
end

%% Sharps, flats

function T = drawSharp(obj, varargin)
% TODO
end

function T = drawDoubleSharp(obj, varargin)
% TODO
end

function T = drawFlat(obj, varargin)
% TODO
end

function T = drawDoubleFlat(obj, varargin)
% TODO
end

function T = drawNatural(obj, varargin)
% TODO
end

%% Bars

function B = drawBar(obj, varargin)
% TODO
end

function R = drawBeginRepeat(obj, varargin)
% TODO
end

function R = drawEndRepeat(obj, varargin)
% TODO
end

%% Pedalling

function P = drawBeginPedal(obj, varargin)
% TODO
end

function P = drawEndPedal(obj, varargin)
% TODO
end

%% Slurs & lines

function S = drawSlur(obj, varargin)
% TODO
% NOTE: You'll have to do this one by hand
end

function P = drawGlissando(obj, varargin)
% TODO
end

%% Multi-measure rest

function R = drawMultiMeasureRest(obj, varargin)
% TODO
end


%% Whole note/rest

function N = drawWholeNote(obj, pitch, varargin)

% TODO

% Possibly transform note
N = obj.transformNote(N, varargin{:});
end

function R = drawWholeRest(obj, varargin)

% TODO

% Possibly transform rest
N = obj.transformNote(N, varargin{:});
end


%% Half note/rest

function N = drawHalfNote(obj, varargin)

% TODO

% Possibly transform note
N = obj.transformNote(N, varargin{:});
end

function R = drawHalfRest(obj, varargin)

% TODO

% Possibly transform rest
N = obj.transformNote(N, varargin{:});
end


%% Quarter note/rest

function N = drawQuarterNote(obj, varargin)

% TODO

% Possibly transform note
N = obj.transformNote(N, varargin{:});
end

function R = drawQuarterRest(obj, varargin)

% TODO

% Possibly transform rest
N = obj.transformNote(N, varargin{:});
end

%% 8th, 16th, 32nd, 64th, ...

function N = drawSingleShortNote(obj, type, varargin)

% TODO

% NOTE: all short notes have a different number of tails. Just store
% the tail as a separate figure, and copy however many times needed.
% Use the quarter note for the head.

% Possibly transform rest
N = obj.transformNote(N, varargin{:});
end

function R = drawShortRest(obj, type, varargin)

% TODO

% NOTE: all short rests are just copies of the eighth rest, so you can
% juse load one image and copy the desired number of times.

% Possibly transform rest
N = obj.transformNote(N, varargin{:});
end


function N = drawShortNoteGroup(obj, types, varargin)

% TODO

% NOTE: Use the quarter note for all the heads. Draw however many
% lines (with "line" command) where needed. Top line should be
% "fatter"; you can do this by adjusting the "linewidth" property

% Possibly transform one or more members of the group
N = obj.transformNote(N, varargin{:});
end


%% Note/Rest transformations
% (dots, accents, inversion, decoration, etc.)

function N = transformNote(obj, N, varargin)

parameters = varargin(1:2:end);
values     = varargin(2:2:end);

if mod(nargin,2)~=0 || ...
~all(cellfun('isclass', parameters), 'char') || ...
~all(cellfun('isclass', values), 'char')
error('transformNotes:no_pv_pairs',...
      'Transforming notes is done by all-text parameter/value pairs.');
end

if numel(parameters)==0
return; end

for ii = 1:numel(parameters)

parameter = lower(parameters{ii});
value     = lower(values{ii});

switch parameter

% Note may be flipped
case 'orientation'
switch value
case {'upright' 'normal'}
% No action
case {'flip', 'flipped', 'upside down'}
N = flipdim(N,1);
otherwise
% error
end

% Duration of note may be extended
case {'extend' 'extension'}
switch value
case {'single' 'dot'}
case {'double' 'dotdot' 'dot dot' 'ddot'}
case {'triple' 'dotdotdot' 'dot dot dot' 'dddot'}
otherwise
% error
end


% Note may be accented
case {'accent' 'accented'}
switch value
case 'portato'
case 'staccato'
case 'staccatissimo'
case 'legato'
case 'marcato'
case 'marcatissimo'
case 'tenuto'
otherwise
% error
end


% Note may be decorated
case {'decoration' 'decorated'}
switch value
case {'thril' 'thriller'}
case {'pralthril' 'pralthriller' 'praller'}
case 'mordent'
case 'arpeggio'
case 'gruppetto'
case 'glissando'
case 'portamento'
case 'schleifer'
case {'grace note' 'appoggiatura'}
case {'striped grace note' 'acciaccatura'}
otherwise
% error
end

otherwise
warning('transformNotes:unsupported_parameter',...
        'Unknown parameter: ''%s''. Ignoring...', parameter);
end
end

end % transformNote

end % Private methods

end % Class definition