\ProvidesClass{memo}[2020/01/07 v0.1 Technical Memorandum Communication]
\NeedsTeXFormat{LaTeX2e}[1995/12/01]

\LoadClass[12pt]{article}

\RequirePackage[utf8]{inputenc}
\RequirePackage[T1]{fontenc}
\RequirePackage[letterpaper, margin=1.0in]{geometry}
\RequirePackage{caption}
    \captionsetup[figure]{labelfont={bf}, labelsep=period}
    \captionsetup[table]{labelfont={bf}, labelsep=period}

\RequirePackage{graphicx}
\RequirePackage{booktabs}
\RequirePackage{subfig}
\RequirePackage{fancyhdr}
\RequirePackage{ifthen}

\setlength{\jot}{1.5em}       % Spacing between gathered equations
\setlength{\parindent}{0em}   % No indent paragraphs
\setlength{\parskip}{1em}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Title formatting
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\renewcommand{\maketitle}{
    \begin{center}
        { \Huge {\bfseries {\scshape \@title}} } \\[1em]
        { \large {\@author} } \\[0.5em]
        { \large {\@date} } \vspace{-1em}
    \end{center}
    \noindent\rule{\linewidth}{1.5pt}
    \par
    \pagestyle{mainContent}
}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Page Numbering
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\fancypagestyle{mainContent}{
    \fancyhead{}\fancyfoot{}
    \renewcommand{\headrulewidth}{0.75pt}
    \fancyfoot[C]{\ifthenelse{\value{page}=1}{}{\thepage}}
}