%Module bughunt

%Import QtCore/QtCoremod.sip
%Import QtGui/QtGuimod.sip
%Import QtWidgets/QtWidgetsmod.sip

class BugHunt
  : public QDialog
{
%TypeHeaderCode
#include "bughunt.h"
%End

public:
    explicit BugHunt(QWidget* parent=nullptr);

/*protected:
    QPushButton* m_leftButton;
    QPushButton* m_rightButton;*/

protected slots:
    void onLeftButton() /HoldGIL/;
    void onRightButton() /HoldGIL/;
};
