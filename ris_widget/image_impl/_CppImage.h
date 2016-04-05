// The MIT License (MIT)
//
// Copyright (c) 2016 WUSTL ZPLAB
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
// Authors: Erik Hvatum <ice.rikh@gmail.com

#pragma once
#include <atomic>
#include <QtCore>

class _CppImage
  : public QObject
{
public:
    enum Status
    {
        Empty,
        AsyncLoading,
        AsyncLoadingFailed,
        Valid
    };

private:
    Q_OBJECT
    Q_ENUM(Status)
    Q_PROPERTY(QString name READ objectName WRITE setObjectName NOTIFY name_changed FINAL)
    Q_PROPERTY(Status status READ get_status NOTIFY status_changed FINAL)
    Q_PROPERTY(std::uint64_t serial READ get_serial NOTIFY serial_changed FINAL)

public:
    explicit _CppImage(const QString& name=QString(), QObject* parent=nullptr);
    virtual ~_CppImage();

    const Status& get_status() const;
    const quint64& get_serial() const;

signals:
    void name_changed(_CppImage*);
    void status_changed(_CppImage*);
    void serial_changed(_CppImage*);

protected:
    static volatile std::atomic<quint64> sm_next_serial;
    static quint64 generate_serial();

    Status m_status;
    quint64 m_serial;
};
