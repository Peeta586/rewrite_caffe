#ifdef USE_LMDB
#ifndef CAFFE_UTIL_DB_LMDB_HPP
#define CAFFE_UTIL_DB_LMDB_HPP

#include <string> 
#include <vector> 

#include "lmdb.h"

#include "caffe/util/db.hpp"

namespace caffe {
namespace db {
    inline void MDB_CHECK(int mdb_status){
        CHECK_EQ(mdb_status, MDB_SUCCESS) << mdb_strerror(mdb_status);
    }

    class LMDBCursor : public Cursor {
        public: 
        /**
         * mdb_env是整个数据库环境的句柄，mdb_dbi是环境中一个数据库的句柄，
         * mdb_key和mdb_data用来存放向数据库中输入数据的“值”。
         * mdb_txn是数据库事物操作的句柄，”txn”是”transaction”的缩写
         */
        explicit LMDBCursor(MDB_txn* mdb_txn, MDB_cursor* mdb_cursor): 
            mdb_txn_(mdb_txn), mdb_cursor_(mdb_cursor), valid_(false) {
                SeekToFirst();
        }

        virtual ~LMDBCursor() {
            mdb_cursor_close(mdb_cursor_);
            mdb_txn_abort(mdb_txn_); // The transaction handle may be discarded using mdb_txn_abort
        }

        virtual void SeekToFirst() { Seek(MDB_FIRST);}
        virtual void Next() { Seek(MDB_NEXT); }
        virtual string key(){
            return string(static_cast<const char*>(mdb_key_.mv_data), mdb_key_.mv_size);
        }
        virtual string value(){
            return string(static_cast<const char*>(mdb_value_.mv_data), mdb_value_.mv_size);
        }

        virtual bool valid() { return valid_; }

        private: 
        void Seek(MDB_cursor_op op){
            int mdb_status = mdb_cursor_get(mdb_cursor_, &mdb_key_, &mdb_value_, op);
            if(mdb_status == MDB_NOTFOUND){ // 如果seek有效，valid被设置为true
                valid_ = false;
            }else {
                MDB_CHECK(mdb_status);
                valid_ = true;
            }
        }
        MDB_txn* mdb_txn_; // 数据库事物
        MDB_cursor* mdb_cursor_;  // 数据库游标
        MDB_val mdb_key_, mdb_value_;  // 当前游标操作的数据库的值
        bool valid_;  // 用于指示这次操作是否成功的。
    };

    class LMDBTransaction : public Transaction {
        public: 
        explicit LMDBTransaction(MDB_env* mdb_env): mdb_env_(mdb_env){ }
        virtual void Put(const string& key, const string& value);
        virtual void Commit();

        private: 
        MDB_env* mdb_env_;
        vector<string> keys, values;
        void DoubleMapSize();

        DISABLE_COPY_AND_ASSIGN(LMDBTransaction);
    };

    class LMDB: public DB{
        public: 
        LMDB() : mdb_env_(NULL){ }
        virtual ~LMDB(){ Close();}
        virtual void Open(const string& source, Mode mode);
        virtual void Close(){
            if(mdb_env_ != NULL){
                mdb_dbi_close(mdb_env_, mdb_dbi_);
                mdb_env_close(mdb_env_);
                mdb_env_ = NULL;
            }
        }
        virtual LMDBCursor* NewCursor();
        virtual LMDBTransaction* NewTransaction();

        private: 
        // mdb_env_create它的作用就是创建一个LMDB的环境handle，就是指向一个分配了内存的地址啦，这个函数会为MDB_env结构体分配一个内存。
        // 它我们使用handle  的时候，
        // 首先要用mdb_env_open()函数打开，最后也要调用mdb_env_close()函数释放掉内存并discard handle。

        MDB_env* mdb_env_; 
        // 在制定环境下的指定数据库用MDB_dbi 句柄去操作
        MDB_dbi mdb_dbi_; // A handle for an individual database in the DB environment
    }

} // namespace db
} // namespace caffe



#endif // CAFFE_UTIL_DB_LMDB_HPP
#endif // USE LMDB