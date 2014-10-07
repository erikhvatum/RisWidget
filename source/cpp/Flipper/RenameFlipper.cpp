// The MIT License (MIT)
// 
// Copyright (c) 2014 WUSTL ZPLAB
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// 
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
// 
// Authors: Erik Hvatum

#include "Common.h"
#include "Flipper.h"
#include "RenameFlipper.h"
#include "RisWidget.h"

RenameFlipper::RenameFlipper(QWidget* parent, Flipper* flipper, RisWidget* rw)
  : QDialog(parent),
    m_flipper(flipper),
    m_rw(rw)
{
    setupUi(this);
    m_flipperNameEdit->setText(m_flipper->getFlipperName());
    m_flipperNameEdit->selectAll();
    connect(m_flipper, &Flipper::flipperNameChanged, this, &RenameFlipper::flipperNameChanged);
}

void RenameFlipper::done(int r)
{
    if(r == QDialog::Accepted)
    {
        QString newName = m_flipperNameEdit->text();
        if(newName != m_flipper->getFlipperName() && m_rw->hasFlipper(newName))
        {
            QMessageBox::information(this, "RisWidget Flipbook Name Conflict", QString("There is already a flipbook named \"%1\".").arg(newName));
        }
        else
        {
            disconnect(m_flipper, &Flipper::flipperNameChanged, this, &RenameFlipper::flipperNameChanged);
            m_flipper->setFlipperName(newName);
            QDialog::done(r);
        }
    }
    else
    {
        disconnect(m_flipper, &Flipper::flipperNameChanged, this, &RenameFlipper::flipperNameChanged);
        QDialog::done(r);
    }
}

void RenameFlipper::flipperNameChanged(Flipper* flipper, QString oldName)
{
    m_flipperNameEdit->setText(flipper->getFlipperName());
    m_flipperNameEdit->selectAll();
}
